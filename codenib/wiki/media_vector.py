# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Semantic vector sidecar for multimodal wiki knowledge."""

from __future__ import annotations

import hmac
import io
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from .._bounded_json import validate_bounded_json_stream, validate_json_complexity
from ._safe_file_reads import read_regular_bytes
from .media_knowledge import MULTIMODAL_KNOWLEDGE_SCHEMA, MULTIMODAL_KNOWLEDGE_VERSION

VISUAL_VECTOR_INDEX_SCHEMA = "codenib.visual-vector-index.v1"
VISUAL_VECTOR_INDEX_VERSION = 1
DEFAULT_VISUAL_VECTOR_DIMENSIONS = 64
DEFAULT_VISUAL_VECTOR_MODEL = "local/hash-visual-embedding-v1"
DEFAULT_VISUAL_VECTOR_PROVIDER = "local"

_MAX_INDEX_BYTES = 128 * 1024 * 1024
_MAX_INDEX_NODES = 2_000_000
_MAX_INDEX_TOKENS = 4_000_000
_MAX_RECORDS = 32_768
_MAX_TEXT_BYTES = 8_192
_MAX_DIMENSIONS = 4_096
_MAX_VECTOR_VALUES = 8_388_608
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")

VisualEmbeddingDocument = Mapping[str, str]
VisualDocumentEmbedder = Callable[
    [Sequence[VisualEmbeddingDocument]], Sequence[Sequence[float]]
]
VisualTextEmbedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def build_visual_vector_index(
    knowledge_view: Mapping[str, Any],
    *,
    previous_index: Mapping[str, Any] | None = None,
    document_embedder: VisualDocumentEmbedder | None = None,
    provider: str = DEFAULT_VISUAL_VECTOR_PROVIDER,
    model: str = DEFAULT_VISUAL_VECTOR_MODEL,
    model_revision: str = "",
    dimensions: int = DEFAULT_VISUAL_VECTOR_DIMENSIONS,
    document_modalities: Sequence[str] = ("text",),
    query_modality: str = "text",
) -> dict[str, Any]:
    """Embed multimodal wiki entries into a stable, queryable sidecar index."""

    embedding_policy = _embedding_policy(
        provider=provider,
        model=model,
        model_revision=model_revision,
        dimensions=dimensions,
        document_modalities=document_modalities,
        query_modality=query_modality,
    )
    embedding_policy_sha256 = _sha256_json(embedding_policy)
    previous_records = _reusable_records(
        previous_index,
        embedding_policy_sha256=embedding_policy_sha256,
        dimensions=dimensions,
    )
    entries = _knowledge_entries(knowledge_view)
    _validate_vector_capacity(len(entries), dimensions=dimensions)
    pending_records: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    reused = 0

    for entry in entries:
        record = _record_without_embedding(entry)
        reusable = previous_records.get(record["artifact_path"])
        if reusable and reusable.get("entry_sha256") == record["entry_sha256"]:
            records.append(dict(reusable))
            reused += 1
            continue
        pending_records.append(record)

    if pending_records:
        vectors = _embed_documents(
            pending_records,
            document_embedder=document_embedder,
            dimensions=dimensions,
            provider=provider,
            model=model,
        )
        for record, vector in zip(pending_records, vectors, strict=True):
            record["embedding"] = vector
            record["embedding_sha256"] = _sha256_json(vector)
            records.append(record)

    records.sort(key=lambda item: item["artifact_path"])
    payload = {
        "schema": VISUAL_VECTOR_INDEX_SCHEMA,
        "version": VISUAL_VECTOR_INDEX_VERSION,
        "knowledge_view_sha256": _digest(
            knowledge_view.get("view_sha256"),
            label="knowledge_view.view_sha256",
        ),
        "embedding_policy": embedding_policy,
        "embedding_policy_sha256": embedding_policy_sha256,
        "entry_count": len(records),
        "reused_record_count": reused,
        "embedded_record_count": len(records) - reused,
        "records": records,
    }
    payload["index_sha256"] = _sha256_json(
        {key: value for key, value in payload.items() if key != "index_sha256"}
    )
    return validate_visual_vector_index(payload)


def search_visual_vector_index(
    index: Mapping[str, Any],
    query: str,
    *,
    limit: int = 5,
    embedder: VisualTextEmbedder | None = None,
) -> list[dict[str, Any]]:
    """Search visual wiki knowledge using the sidecar vector view."""

    validated = validate_visual_vector_index(index)
    query_text = _bounded_text(query, label="query")
    if not query_text:
        return []
    limit = _limit(limit)
    policy = validated["embedding_policy"]
    query_vector = _embed_texts(
        [query_text],
        embedder=embedder,
        dimensions=policy["dimensions"],
        provider=policy["provider"],
        model=policy["model"],
    )[0]
    results = []
    for record in validated["records"]:
        score = _dot(query_vector, record["embedding"])
        if score <= 0:
            continue
        results.append(
            {
                "artifact_path": record["artifact_path"],
                "score": round(score, 6),
                "caption": record.get("caption", ""),
                "role_hint": record.get("role_hint", ""),
                "mime_type": record.get("mime_type", ""),
                "source_paths": list(record.get("source_paths") or ()),
                "symbols": list(record.get("symbols") or ()),
                "entry_sha256": record["entry_sha256"],
            }
        )
    results.sort(key=lambda item: (-item["score"], item["artifact_path"]))
    return results[:limit]


def save_visual_vector_index(index: Mapping[str, Any], path: str | Path) -> None:
    """Atomically write a validated visual vector sidecar index."""

    validated = validate_visual_vector_index(index)
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            validated,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    if len(payload) > _MAX_INDEX_BYTES:
        raise ValueError("visual vector index exceeds the byte limit")
    try:
        existing = os.stat(destination, follow_symlinks=False)
    except FileNotFoundError:
        existing_mode = None
    else:
        existing_mode = (
            stat.S_IMODE(existing.st_mode) if stat.S_ISREG(existing.st_mode) else None
        )
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            if existing_mode is not None:
                os.fchmod(handle.fileno(), existing_mode)
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def load_visual_vector_index(path: str | Path) -> dict[str, Any]:
    """Load and validate a persisted visual vector sidecar index."""

    source = Path(path).expanduser()
    raw = read_regular_bytes(source, max_bytes=_MAX_INDEX_BYTES)
    if raw is None:
        raise ValueError(
            "visual vector index must be a stable regular file within the byte limit"
        )
    validate_bounded_json_stream(
        io.BytesIO(raw),
        label="visual vector index",
        max_bytes=_MAX_INDEX_BYTES,
        max_nodes=_MAX_INDEX_NODES,
        max_lexical_tokens=_MAX_INDEX_TOKENS,
    )
    try:
        data = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_nonfinite_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("visual vector index contains invalid JSON") from exc
    return validate_visual_vector_index(data)


def validate_visual_vector_index(index: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized visual vector index or raise ``ValueError``."""

    if not isinstance(index, Mapping):
        raise ValueError("visual vector index must be an object")
    data = dict(index)
    if data.get("schema") != VISUAL_VECTOR_INDEX_SCHEMA:
        raise ValueError("visual vector index schema is unsupported")
    if (
        type(data.get("version")) is not int
        or data["version"] != VISUAL_VECTOR_INDEX_VERSION
    ):
        raise ValueError("visual vector index version is unsupported")
    policy = _mapping(data.get("embedding_policy"), label="embedding_policy")
    dimensions = _dimensions(policy.get("dimensions"))
    normalized_policy = _embedding_policy(
        provider=_nonempty_text(policy.get("provider"), label="embedding provider"),
        model=_nonempty_text(policy.get("model"), label="embedding model"),
        model_revision=_bounded_text(
            policy.get("model_revision"),
            label="embedding model revision",
            allow_empty=True,
        ),
        dimensions=dimensions,
        document_modalities=_modality_list(policy.get("document_modalities")),
        query_modality=_modality(
            policy.get("query_modality"),
            label="query modality",
        ),
    )
    if policy != normalized_policy:
        raise ValueError("visual vector embedding policy is not canonical")
    policy_sha256 = _digest(
        data.get("embedding_policy_sha256"),
        label="embedding_policy_sha256",
    )
    if not hmac.compare_digest(policy_sha256, _sha256_json(normalized_policy)):
        raise ValueError("visual vector embedding policy hash does not match")
    records = _record_list(data.get("records"), dimensions=dimensions)
    _validate_vector_capacity(len(records), dimensions=dimensions)
    if type(data.get("entry_count")) is not int or data["entry_count"] != len(records):
        raise ValueError("visual vector index entry_count is invalid")
    for key in ("reused_record_count", "embedded_record_count"):
        if type(data.get(key)) is not int or data[key] < 0:
            raise ValueError(f"visual vector index {key} is invalid")
    if data["reused_record_count"] + data["embedded_record_count"] != len(records):
        raise ValueError("visual vector index reuse accounting is invalid")
    data.update(
        {
            "knowledge_view_sha256": _digest(
                data.get("knowledge_view_sha256"),
                label="knowledge_view_sha256",
            ),
            "embedding_policy": normalized_policy,
            "records": records,
        }
    )
    expected = _sha256_json(
        {key: value for key, value in data.items() if key != "index_sha256"}
    )
    recorded = _digest(data.get("index_sha256"), label="index_sha256")
    if not hmac.compare_digest(recorded, expected):
        raise ValueError("visual vector index hash does not match")
    validate_json_complexity(
        data,
        label="visual vector index",
        max_nodes=_MAX_INDEX_NODES,
    )
    if len(_canonical_json_bytes(data)) > _MAX_INDEX_BYTES:
        raise ValueError("visual vector index exceeds the byte limit")
    return data


def deterministic_visual_text_embeddings(
    texts: Sequence[str],
    *,
    dimensions: int = DEFAULT_VISUAL_VECTOR_DIMENSIONS,
    seed: str = DEFAULT_VISUAL_VECTOR_MODEL,
) -> list[list[float]]:
    """Return deterministic local embeddings for offline previews and tests."""

    dimensions = _dimensions(dimensions)
    vectors = []
    for text in texts:
        vector = [0.0] * dimensions
        tokens = _tokens(_bounded_text(text, label="embedding text", allow_empty=True))
        for token in tokens:
            digest = sha256(f"{seed}\0{token}".encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % dimensions
            sign = -1.0 if digest[4] & 1 else 1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [round(value / norm, 8) for value in vector]
        vectors.append(vector)
    return vectors


def _record_without_embedding(entry: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _mapping(entry.get("artifact"), label="entry.artifact")
    facts = _mapping(entry.get("facts"), label="entry.facts")
    bindings = [
        _mapping(binding, label="entry.binding")
        for binding in _iter_limited(entry.get("bindings"), limit=256)
    ]
    artifact_path = _relative_path(artifact.get("path"), label="artifact.path")
    source_paths = sorted(
        {
            _relative_path(binding.get("source_path"), label="binding.source_path")
            for binding in bindings
            if binding.get("source_path")
        }
    )
    symbols = sorted(
        {
            _bounded_text(binding.get("symbol"), label="binding.symbol")
            for binding in bindings
            if binding.get("symbol")
        }
    )
    embedding_text = _entry_embedding_text(artifact, facts, bindings, entry)
    entry_payload = {
        "artifact": artifact,
        "facts": facts,
        "bindings": bindings,
        "embedding_text": embedding_text,
    }
    return {
        "artifact_path": artifact_path,
        "artifact_sha256": _optional_digest(
            artifact.get("sha256"),
            label="artifact.sha256",
        ),
        "caption": _bounded_text(
            artifact.get("caption"),
            label="artifact.caption",
            allow_empty=True,
        ),
        "role_hint": _bounded_text(
            artifact.get("role_hint"),
            label="artifact.role_hint",
            allow_empty=True,
        ),
        "mime_type": _bounded_text(
            artifact.get("mime_type"),
            label="artifact.mime_type",
            allow_empty=True,
        ),
        "source_paths": source_paths,
        "symbols": symbols,
        "embedding_text": embedding_text,
        "entry_sha256": _sha256_json(entry_payload),
    }


def _entry_embedding_text(
    artifact: Mapping[str, Any],
    facts: Mapping[str, Any],
    bindings: Sequence[Mapping[str, Any]],
    entry: Mapping[str, Any],
) -> str:
    parts: list[str] = [
        _bounded_text(artifact.get("path"), label="artifact.path", allow_empty=True),
        _bounded_text(
            artifact.get("role_hint"),
            label="artifact.role_hint",
            allow_empty=True,
        ),
        _bounded_text(
            artifact.get("caption"),
            label="artifact.caption",
            allow_empty=True,
        ),
        _bounded_text(
            artifact.get("surrounding_text"),
            label="artifact.surrounding_text",
            allow_empty=True,
        ),
        _bounded_text(
            entry.get("search_text"),
            label="entry.search_text",
            allow_empty=True,
        ),
    ]
    for entity in _iter_limited(facts.get("entities"), limit=256):
        if isinstance(entity, Mapping):
            parts.extend(
                _bounded_text(entity.get(key), label=f"entity.{key}", allow_empty=True)
                for key in ("name", "type", "evidence")
            )
    for claim in _iter_limited(facts.get("claims"), limit=256):
        if isinstance(claim, Mapping):
            parts.extend(
                _bounded_text(claim.get(key), label=f"claim.{key}", allow_empty=True)
                for key in ("text", "evidence")
            )
    for binding in bindings:
        parts.extend(
            _bounded_text(binding.get(key), label=f"binding.{key}", allow_empty=True)
            for key in ("entity_name", "source_path", "symbol", "evidence")
        )
    return _bounded_text(
        " ".join(part for part in parts if part), label="embedding text"
    )


def _embed_documents(
    records: Sequence[Mapping[str, Any]],
    *,
    document_embedder: VisualDocumentEmbedder | None,
    dimensions: int,
    provider: str,
    model: str,
) -> list[list[float]]:
    documents = [
        {
            "artifact_path": str(record["artifact_path"]),
            "artifact_sha256": str(record["artifact_sha256"]),
            "mime_type": str(record["mime_type"]),
            "text": str(record["embedding_text"]),
        }
        for record in records
    ]
    if document_embedder is None:
        if provider != DEFAULT_VISUAL_VECTOR_PROVIDER:
            raise ValueError(
                "non-local visual vector indexes require a document embedder"
            )
        raw_vectors = deterministic_visual_text_embeddings(
            [document["text"] for document in documents],
            dimensions=dimensions,
            seed=model,
        )
    else:
        raw_vectors = document_embedder(documents)
    vectors = [
        _normalize_vector(vector, dimensions=dimensions, label=f"embedding[{index}]")
        for index, vector in enumerate(raw_vectors)
    ]
    if len(vectors) != len(documents):
        raise ValueError(
            "visual document embedder returned the wrong number of vectors"
        )
    return vectors


def _embed_texts(
    texts: Sequence[str],
    *,
    embedder: VisualTextEmbedder | None,
    dimensions: int,
    provider: str,
    model: str,
) -> list[list[float]]:
    if embedder is None:
        if provider != DEFAULT_VISUAL_VECTOR_PROVIDER:
            raise ValueError("non-local visual vector indexes require an embedder")
        raw_vectors = deterministic_visual_text_embeddings(
            texts,
            dimensions=dimensions,
            seed=model,
        )
    else:
        raw_vectors = embedder(texts)
    vectors = [
        _normalize_vector(vector, dimensions=dimensions, label=f"embedding[{index}]")
        for index, vector in enumerate(raw_vectors)
    ]
    if len(vectors) != len(texts):
        raise ValueError("visual vector embedder returned the wrong number of vectors")
    return vectors


def _normalize_vector(
    vector: Sequence[float],
    *,
    dimensions: int,
    label: str,
) -> list[float]:
    if isinstance(vector, (str, bytes, bytearray)):
        raise ValueError(f"{label} has invalid dimensions")
    try:
        values_in = list(vector)
    except TypeError as exc:
        raise ValueError(f"{label} must be a numeric vector") from exc
    if len(values_in) != dimensions:
        raise ValueError(f"{label} has invalid dimensions")
    values = []
    for value in values_in:
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite value")
        values.append(float(value))
    norm = math.sqrt(sum(value * value for value in values))
    if not norm:
        return [0.0] * dimensions
    return [round(value / norm, 8) for value in values]


def _reusable_records(
    previous_index: Mapping[str, Any] | None,
    *,
    embedding_policy_sha256: str,
    dimensions: int,
) -> dict[str, dict[str, Any]]:
    if previous_index is None:
        return {}
    try:
        validated = validate_visual_vector_index(previous_index)
    except ValueError:
        return {}
    if validated["embedding_policy_sha256"] != embedding_policy_sha256:
        return {}
    records = {}
    for record in validated["records"]:
        try:
            normalized = dict(record)
            normalized["embedding"] = _normalize_vector(
                normalized.get("embedding") or (),
                dimensions=dimensions,
                label="previous embedding",
            )
            normalized["embedding_sha256"] = _sha256_json(normalized["embedding"])
            records[record["artifact_path"]] = normalized
        except ValueError:
            return {}
    return records


def _knowledge_entries(view: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not isinstance(view, Mapping):
        raise ValueError("knowledge view must be an object")
    if view.get("schema") != MULTIMODAL_KNOWLEDGE_SCHEMA:
        raise ValueError("knowledge view schema is unsupported")
    if (
        type(view.get("version")) is not int
        or view["version"] != MULTIMODAL_KNOWLEDGE_VERSION
    ):
        raise ValueError("knowledge view version is unsupported")
    recorded_sha256 = _digest(
        view.get("view_sha256"),
        label="knowledge_view.view_sha256",
    )
    expected_sha256 = _sha256_json(
        {key: value for key, value in view.items() if key != "view_sha256"}
    )
    if not hmac.compare_digest(recorded_sha256, expected_sha256):
        raise ValueError("knowledge view hash does not match")
    entries = list(_iter_limited(view.get("entries"), limit=_MAX_RECORDS))
    if type(view.get("entry_count")) is not int or view["entry_count"] != len(entries):
        raise ValueError("knowledge view entry_count is invalid")
    if any(not isinstance(entry, Mapping) for entry in entries):
        raise ValueError("knowledge view entries must be objects")
    return entries


def _embedding_policy(
    *,
    provider: str,
    model: str,
    model_revision: str,
    dimensions: int,
    document_modalities: Sequence[str],
    query_modality: str,
) -> dict[str, Any]:
    return {
        "provider": _nonempty_text(provider, label="embedding provider"),
        "model": _nonempty_text(model, label="embedding model"),
        "model_revision": _bounded_text(
            model_revision,
            label="embedding model revision",
            allow_empty=True,
        ),
        "dimensions": _dimensions(dimensions),
        "normalized": True,
        "input_contract": "codenib.visual-knowledge-entry.v1",
        "document_modalities": _modality_list(document_modalities),
        "query_modality": _modality(query_modality, label="query modality"),
    }


def _record_list(value: Any, *, dimensions: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for item in _iter_limited(value, limit=_MAX_RECORDS):
        record = _mapping(item, label="record")
        artifact_path = _relative_path(
            record.get("artifact_path"), label="artifact_path"
        )
        if artifact_path in seen_paths:
            raise ValueError(f"duplicate visual vector record path: {artifact_path}")
        seen_paths.add(artifact_path)
        embedding = _normalize_vector(
            record.get("embedding") or (),
            dimensions=dimensions,
            label=f"embedding for {artifact_path}",
        )
        embedding_sha256 = _digest(
            record.get("embedding_sha256"),
            label="record.embedding_sha256",
        )
        if not hmac.compare_digest(embedding_sha256, _sha256_json(embedding)):
            raise ValueError("visual vector record embedding hash does not match")
        records.append(
            {
                "artifact_path": artifact_path,
                "artifact_sha256": _optional_digest(
                    record.get("artifact_sha256"),
                    label="artifact_sha256",
                ),
                "caption": _bounded_text(
                    record.get("caption"),
                    label="caption",
                    allow_empty=True,
                ),
                "role_hint": _bounded_text(
                    record.get("role_hint"),
                    label="role_hint",
                    allow_empty=True,
                ),
                "mime_type": _bounded_text(
                    record.get("mime_type"),
                    label="mime_type",
                    allow_empty=True,
                ),
                "source_paths": sorted(
                    _relative_path(path, label="source_path")
                    for path in _iter_limited(record.get("source_paths"), limit=256)
                ),
                "symbols": sorted(
                    _bounded_text(symbol, label="symbol")
                    for symbol in _iter_limited(record.get("symbols"), limit=256)
                ),
                "embedding_text": _bounded_text(
                    record.get("embedding_text"),
                    label="embedding_text",
                    allow_empty=True,
                ),
                "entry_sha256": _digest(
                    record.get("entry_sha256"),
                    label="record.entry_sha256",
                ),
                "embedding": embedding,
                "embedding_sha256": embedding_sha256,
            }
        )
    records.sort(key=lambda item: item["artifact_path"])
    return records


def _iter_limited(value: Any, *, limit: int) -> Iterable[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return ()
    try:
        values = list(value or ())
    except TypeError:
        return ()
    if len(values) > limit:
        raise ValueError("visual vector collection exceeds the item limit")
    return values


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _dimensions(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_DIMENSIONS:
        raise ValueError("visual vector dimensions are invalid")
    return value


def _modality_list(value: Any) -> list[str]:
    modalities = sorted(
        {
            _modality(item, label="document modality")
            for item in _iter_limited(value, limit=8)
        }
    )
    if not modalities:
        raise ValueError("at least one document modality is required")
    return modalities


def _modality(value: Any, *, label: str) -> str:
    modality = _nonempty_text(value, label=label)
    if modality not in {
        "image",
        "interleaved",
        "text",
        "video",
        "visual_document",
    }:
        raise ValueError(f"{label} is unsupported")
    return modality


def _validate_vector_capacity(record_count: int, *, dimensions: int) -> None:
    if record_count * dimensions > _MAX_VECTOR_VALUES:
        raise ValueError("visual vector index exceeds the vector value limit")


def _limit(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= 50:
        raise ValueError("limit must be between 1 and 50")
    return value


def _nonempty_text(value: Any, *, label: str) -> str:
    text = _bounded_text(value, label=label)
    if not text:
        raise ValueError(f"{label} is required")
    return text


def _bounded_text(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if value is None and allow_empty:
        return ""
    if type(value) is not str:
        raise ValueError(f"{label} must be a string")
    text = value.strip()
    if not text and not allow_empty:
        raise ValueError(f"{label} is required")
    if len(text.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise ValueError(f"{label} exceeds the byte limit")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in text):
        raise ValueError(f"{label} contains control characters")
    return text


def _relative_path(value: Any, *, label: str) -> str:
    text = _bounded_text(value, label=label)
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


def _tokens(value: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(value) if len(token) >= 2]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _digest(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 digest")
    return value


def _optional_digest(value: Any, *, label: str) -> str:
    if value is None or value == "":
        return ""
    return _digest(value, label=label)


def _sha256_json(payload: Any) -> str:
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValueError(
            "visual vector index must contain bounded JSON values"
        ) from exc


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_number(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


__all__ = [
    "DEFAULT_VISUAL_VECTOR_DIMENSIONS",
    "DEFAULT_VISUAL_VECTOR_MODEL",
    "DEFAULT_VISUAL_VECTOR_PROVIDER",
    "VISUAL_VECTOR_INDEX_SCHEMA",
    "VISUAL_VECTOR_INDEX_VERSION",
    "VisualDocumentEmbedder",
    "VisualEmbeddingDocument",
    "VisualTextEmbedder",
    "build_visual_vector_index",
    "deterministic_visual_text_embeddings",
    "load_visual_vector_index",
    "save_visual_vector_index",
    "search_visual_vector_index",
    "validate_visual_vector_index",
]
