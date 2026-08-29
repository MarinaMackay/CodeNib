# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""WeMM-Embedding backend for visual wiki retrieval."""

from __future__ import annotations

import hashlib
import hmac
import os
import tempfile
from collections.abc import Mapping, Sequence
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Any

from ..index.embedding.model_policy import resolve_embedding_load_policy
from ._safe_file_reads import read_regular_bytes
from .media_vector import VisualEmbeddingDocument

WEMM_VISUAL_VECTOR_PROVIDER = "wemm/sentence-transformers"
DEFAULT_WEMM_MODEL = "tencent/WeMM-Embedding-2B"
DEFAULT_WEMM_DIMENSIONS = 256

_MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
_MAX_BATCH_BYTES = 128 * 1024 * 1024
_MAX_BATCH_SIZE = 64
_MAX_TEXT_BYTES = 8_192
_RASTER_MIME_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


class WeMMVisualEmbeddingBackend:
    """Encode repository images and text queries in WeMM's shared space.

    Remote-code execution is disabled unless the caller opts in. CodeNib's
    embedding policy additionally requires a full immutable revision for a
    remote model whenever ``trust_remote_code`` is enabled.
    """

    provider = WEMM_VISUAL_VECTOR_PROVIDER
    document_modalities = ("image", "text")
    query_modality = "text"

    def __init__(
        self,
        *,
        repo_path: str | Path,
        model: str = DEFAULT_WEMM_MODEL,
        dimensions: int = DEFAULT_WEMM_DIMENSIONS,
        revision: str | None = None,
        trust_remote_code: bool = False,
        device: str | None = None,
        batch_size: int = 1,
        model_instance: Any | None = None,
    ) -> None:
        self.repo_path = _repository_root(repo_path)
        self.model_name = _nonempty_text(model, label="WeMM model", max_bytes=4096)
        self.dimensions = _dimensions(dimensions)
        self.batch_size = _batch_size(batch_size)
        self.device = _optional_text(device, label="WeMM device", max_bytes=256)
        self.load_policy = resolve_embedding_load_policy(
            self.model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        self.model_revision = self.load_policy.revision or ""
        if (
            model_instance is None
            and not Path(self.model_name).expanduser().is_dir()
            and not self.load_policy.trust_remote_code
        ):
            raise ValueError(
                "remote WeMM models require trust_remote_code=True and an "
                "immutable revision"
            )
        self._model = model_instance
        self._model_contract_checked = False

    def embed_documents(
        self,
        documents: Sequence[VisualEmbeddingDocument],
    ) -> list[list[float]]:
        """Embed verified raster artifacts, with grounded text for SVG fallback."""

        normalized = _documents(documents)
        vectors: list[list[float]] = []
        for start in range(0, len(normalized), self.batch_size):
            batch = normalized[start : start + self.batch_size]
            vectors.extend(self._embed_document_batch(batch))
        return vectors

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed bounded text queries in the same WeMM space as documents."""

        queries = [
            _nonempty_text(text, label="visual query", max_bytes=_MAX_TEXT_BYTES)
            for text in _bounded_sequence(
                texts,
                label="visual queries",
                limit=_MAX_BATCH_SIZE * 512,
            )
        ]
        vectors: list[list[float]] = []
        for start in range(0, len(queries), self.batch_size):
            batch = queries[start : start + self.batch_size]
            vectors.extend(self._encode(batch, method="encode_query"))
        return vectors

    def _embed_document_batch(
        self,
        documents: Sequence[dict[str, str]],
    ) -> list[list[float]]:
        prepared: list[tuple[dict[str, str], bytes]] = []
        total_bytes = 0
        for document in documents:
            payload = self._verified_artifact(document)
            total_bytes += len(payload)
            if total_bytes > _MAX_BATCH_BYTES:
                raise ValueError("WeMM document batch exceeds the byte limit")
            prepared.append((document, payload))

        with tempfile.TemporaryDirectory(prefix="codenib-wemm-") as temp_dir:
            inputs: list[Any] = []
            for index, (document, payload) in enumerate(prepared):
                suffix = _RASTER_MIME_SUFFIXES.get(document["mime_type"])
                if suffix is None:
                    inputs.append(document["text"])
                    continue
                staged = Path(temp_dir) / f"artifact-{index}{suffix}"
                _write_private_bytes(staged, payload)
                inputs.append(
                    {
                        "image": os.fspath(staged),
                        "text": document["text"],
                    }
                )
            return self._encode(inputs, method="encode_document")

    def _verified_artifact(self, document: Mapping[str, str]) -> bytes:
        artifact_path = _relative_path(
            document["artifact_path"],
            label="artifact_path",
        )
        candidate = self.repo_path.joinpath(*PurePosixPath(artifact_path).parts)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.repo_path)
        except (OSError, ValueError) as exc:
            raise ValueError("WeMM artifact must stay inside the repository") from exc
        payload = read_regular_bytes(candidate, max_bytes=_MAX_ARTIFACT_BYTES)
        if payload is None:
            raise ValueError("WeMM artifact must be a stable bounded regular file")
        expected = _sha256(document["artifact_sha256"], label="artifact_sha256")
        if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected):
            raise ValueError("WeMM artifact content hash does not match")
        return payload

    def _encode(self, inputs: Sequence[Any], *, method: str) -> list[list[float]]:
        if not inputs:
            return []
        model = self._loaded_model()
        encoder = getattr(model, method, None)
        if not callable(encoder):
            raise RuntimeError(
                "WeMM requires sentence-transformers with encode_document() "
                "and encode_query() support"
            )
        encoded = encoder(
            list(inputs),
            batch_size=min(self.batch_size, len(inputs)),
            truncate_dim=self.dimensions,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        values = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
        if values and isinstance(values[0], (int, float)):
            values = [values]
        if len(values) != len(inputs):
            raise ValueError("WeMM returned the wrong number of embeddings")
        return [list(vector) for vector in values]

    def _loaded_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "WeMM embedding requires the CodeNib semantic dependencies"
                ) from exc
            options: dict[str, Any] = {
                "trust_remote_code": self.load_policy.trust_remote_code,
            }
            if self.load_policy.revision:
                options["revision"] = self.load_policy.revision
            if self.device:
                options["device"] = self.device
            self._model = SentenceTransformer(self.model_name, **options)
        if not self._model_contract_checked:
            self._validate_model_contract(self._model)
            self._model_contract_checked = True
        return self._model

    def _validate_model_contract(self, model: Any) -> None:
        supported = None
        try:
            supported = model[0].auto_model.config.matryoshka_dimensions
        except (AttributeError, IndexError, KeyError, TypeError):
            pass
        if supported is not None and self.dimensions not in supported:
            raise ValueError(
                "WeMM dimensions are unsupported by the selected model: "
                f"{sorted(supported)}"
            )


def _documents(value: Any) -> list[dict[str, str]]:
    documents = []
    allowed = {"artifact_path", "artifact_sha256", "mime_type", "text"}
    for item in _bounded_sequence(
        value,
        label="visual embedding documents",
        limit=_MAX_BATCH_SIZE * 512,
    ):
        if not isinstance(item, Mapping):
            raise ValueError("visual embedding document must be an object")
        keys = _bounded_sequence(
            item.keys(),
            label="visual embedding document fields",
            limit=len(allowed) + 1,
        )
        if len(keys) != len(allowed) or set(keys) != allowed:
            raise ValueError("visual embedding document fields are invalid")
        documents.append(
            {
                "artifact_path": _relative_path(
                    item.get("artifact_path"),
                    label="artifact_path",
                ),
                "artifact_sha256": _sha256(
                    item.get("artifact_sha256"),
                    label="artifact_sha256",
                ),
                "mime_type": _nonempty_text(
                    item.get("mime_type"),
                    label="artifact MIME type",
                    max_bytes=256,
                ),
                "text": _nonempty_text(
                    item.get("text"),
                    label="visual embedding text",
                    max_bytes=_MAX_TEXT_BYTES,
                ),
            }
        )
    return documents


def _bounded_sequence(value: Any, *, label: str, limit: int) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{label} must be a sequence")
    try:
        iterator = iter(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a sequence") from exc
    items = list(islice(iterator, limit + 1))
    if len(items) > limit:
        raise ValueError(f"{label} exceeds the item limit")
    return items


def _repository_root(value: str | Path) -> Path:
    try:
        root = Path(value).expanduser().resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("WeMM repository root is invalid") from exc
    if not root.is_dir():
        raise ValueError("WeMM repository root must be a directory")
    return root


def _dimensions(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= 4096:
        raise ValueError("WeMM dimensions must be between 1 and 4096")
    return value


def _batch_size(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_BATCH_SIZE:
        raise ValueError(f"WeMM batch size must be between 1 and {_MAX_BATCH_SIZE}")
    return value


def _relative_path(value: Any, *, label: str) -> str:
    text = _nonempty_text(value, label=label, max_bytes=4096)
    if "\\" in text:
        raise ValueError(f"{label} must be a repository-relative path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != text
    ):
        raise ValueError(f"{label} must be a repository-relative path")
    return path.as_posix()


def _sha256(value: Any, *, label: str) -> str:
    text = _nonempty_text(value, label=label, max_bytes=64)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a sha256 digest")
    return text


def _optional_text(value: Any, *, label: str, max_bytes: int) -> str:
    if value is None:
        return ""
    return _nonempty_text(value, label=label, max_bytes=max_bytes)


def _nonempty_text(value: Any, *, label: str, max_bytes: int) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{label} is required")
    if len(text.encode("utf-8")) > max_bytes:
        raise ValueError(f"{label} exceeds the byte limit")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in text):
        raise ValueError(f"{label} contains control characters")
    return text


def _write_private_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("failed to stage WeMM artifact bytes")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "DEFAULT_WEMM_DIMENSIONS",
    "DEFAULT_WEMM_MODEL",
    "WEMM_VISUAL_VECTOR_PROVIDER",
    "WeMMVisualEmbeddingBackend",
]
