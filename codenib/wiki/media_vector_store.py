# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""FAISS materialization for validated multimodal wiki vector indexes."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .media_vector import (
    DEFAULT_VISUAL_VECTOR_PROVIDER,
    VISUAL_VECTOR_INDEX_SCHEMA,
    VisualTextEmbedder,
    deterministic_visual_text_embeddings,
    validate_visual_vector_index,
)

VISUAL_VECTOR_DOCUMENT_TYPE = "visual_knowledge"
VISUAL_VECTOR_STORE_SCHEMA = "codenib.visual-vector-store.v1"
_MAX_QUERY_BYTES = 8_192
_MAX_SEARCH_LIMIT = 50


@dataclass(frozen=True, slots=True)
class VisualVectorDocument:
    """Document shape consumed by :class:`CodeVectorStore`."""

    page_content: str
    metadata: dict[str, Any]


class VisualQueryEmbedding:
    """Expose a validated visual query embedder to ``CodeVectorStore``."""

    def __init__(
        self,
        index: Mapping[str, Any],
        *,
        embedder: VisualTextEmbedder | None = None,
    ) -> None:
        validated = validate_visual_vector_index(index)
        self._policy = validated["embedding_policy"]
        self._embedder = embedder

    def embed_query(self, text: str) -> list[float]:
        text = _query_text(text)
        provider = self._policy["provider"]
        dimensions = self._policy["dimensions"]
        # CodeVectorStore probes its wrapper during construction even though
        # this sidecar already carries a validated, immutable dimension. Do not
        # start a remote or GPU-backed multimodal model for that internal probe.
        if text == "codenib-dimension-probe":
            return [1.0, *([0.0] * (dimensions - 1))]
        if self._embedder is None:
            if provider != DEFAULT_VISUAL_VECTOR_PROVIDER:
                raise ValueError(
                    "non-local visual vector stores require a query embedder"
                )
            vectors = deterministic_visual_text_embeddings(
                [text],
                dimensions=dimensions,
                seed=self._policy["model"],
            )
        else:
            vectors = self._embedder([text])
        try:
            materialized = list(vectors)
        except TypeError as exc:
            raise ValueError("visual query embedder must return one vector") from exc
        if len(materialized) != 1:
            raise ValueError("visual query embedder must return one vector")
        return _normalized_vector(materialized[0], dimensions=dimensions)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        del texts
        raise RuntimeError(
            "visual vector documents must use the precomputed sidecar embeddings"
        )


def visual_vector_store_identity(
    index: Mapping[str, Any],
    *,
    index_type: str = "flat",
) -> dict[str, Any]:
    """Return the stable schema-8 identity for one visual embedding policy."""

    validated = validate_visual_vector_index(index)
    normalized_index_type = _index_type(index_type)
    policy = validated["embedding_policy"]
    policy_sha256 = validated["embedding_policy_sha256"]
    model = f"codenib/visual-{policy_sha256}"
    return {
        "builder_schema": 8,
        "embedding_model": model,
        # The native store receives precomputed vectors through an in-process
        # adapter. The original producer remains explicit below.
        "embedding_provider": "local",
        "embedding_dimension": policy["dimensions"],
        "dimension": policy["dimensions"],
        "index_metric": "ip",
        "index_type": normalized_index_type,
        "visual_vector_store_schema": VISUAL_VECTOR_STORE_SCHEMA,
        "visual_vector_index_schema": VISUAL_VECTOR_INDEX_SCHEMA,
        "visual_embedding_policy": policy,
        "visual_embedding_policy_sha256": policy_sha256,
    }


def create_visual_vector_store(
    index: Mapping[str, Any],
    *,
    query_embedder: VisualTextEmbedder | None = None,
    store_path: str | Path | None = None,
    index_type: str = "flat",
    ivf_nlist: int = 100,
    ivf_nprobe: int = 8,
):
    """Create a CodeNib FAISS store bound to a visual embedding policy."""

    validated = validate_visual_vector_index(index)
    if (
        validated["embedding_policy"]["provider"]
        != DEFAULT_VISUAL_VECTOR_PROVIDER
        and query_embedder is None
    ):
        raise ValueError("non-local visual vector stores require a query embedder")
    identity = visual_vector_store_identity(validated, index_type=index_type)
    from ..index.embedding.vector_store import CodeVectorStore

    return CodeVectorStore(
        embedding_model=identity["embedding_model"],
        embedding_provider=identity["embedding_provider"],
        dimension=identity["dimension"],
        index_type=identity["index_type"],
        index_metric=identity["index_metric"],
        ivf_nlist=ivf_nlist,
        ivf_nprobe=ivf_nprobe,
        store_path=str(store_path) if store_path is not None else None,
        embedding=VisualQueryEmbedding(validated, embedder=query_embedder),
        artifact_metadata=identity,
    )


def visual_vector_documents(
    index: Mapping[str, Any],
) -> tuple[list[VisualVectorDocument], list[Any]]:
    """Translate a sidecar into schema-8 documents and float32 vectors."""

    validated = validate_visual_vector_index(index)
    import numpy as np

    documents: list[VisualVectorDocument] = []
    vectors: list[Any] = []
    for row, record in enumerate(validated["records"]):
        content = record["embedding_text"]
        artifact_path = record["artifact_path"]
        caption = record.get("caption") or artifact_path
        content_hash = hashlib.md5(  # noqa: S324 - vector row compatibility
            content.encode("utf-8", errors="replace"),
            usedforsecurity=False,
        ).hexdigest()
        documents.append(
            VisualVectorDocument(
                page_content=content,
                metadata={
                    "chunk_id": row,
                    "chunk_type": VISUAL_VECTOR_DOCUMENT_TYPE,
                    "name": caption,
                    "file": artifact_path,
                    "start_line": 0,
                    "end_line": 0,
                    "node_id": (
                        f"{artifact_path}:visual:{record['entry_sha256'][:16]}"
                    ),
                    "level": "l2",
                    "content_hash": content_hash,
                },
            )
        )
        vectors.append(np.asarray(record["embedding"], dtype=np.float32))
    return documents, vectors


def update_visual_vector_store(
    store: Any,
    index: Mapping[str, Any],
    *,
    previous_index: Mapping[str, Any] | None = None,
    threshold: float = 0.1,
) -> dict[str, Any]:
    """Incrementally apply a visual sidecar to an existing CodeVectorStore."""

    validated = validate_visual_vector_index(index)
    _validate_threshold(threshold)
    expected_identity = visual_vector_store_identity(
        validated,
        index_type=getattr(store, "index_type", "flat"),
    )
    if dict(getattr(store, "artifact_metadata", {})) != expected_identity:
        raise ValueError("visual vector store identity does not match the sidecar")
    if getattr(store, "dimension", None) != expected_identity["dimension"]:
        raise ValueError("visual vector store dimensions do not match the sidecar")
    if getattr(store, "index_metric", None) != "ip":
        raise ValueError("visual vector stores require inner-product search")

    documents, vectors = visual_vector_documents(validated)
    previous = (
        validate_visual_vector_index(previous_index)
        if previous_index is not None
        else None
    )
    changed_count, changed_hashes = _changed_entries(previous, validated)
    current_documents = list(getattr(store, "l2_documents", ()) or ())
    use_delta = bool(
        previous is not None
        and current_documents
        and getattr(store, "index_type", None) == "flat"
        and changed_count / max(len(documents), 1) <= threshold
    )
    if use_delta:
        store.delta_update(
            documents,
            vectors,
            changed_hashes,
            level="l2",
            # The entry-count policy above decides whether delta is useful.
            # A hash set can be smaller when duplicate content is present.
            threshold=1.0,
        )
        mode = "delta"
    else:
        store.rebuild_from_embeddings(documents, vectors, level="l2")
        mode = "rebuild"
    _canonicalize_store_rows(store)
    return {
        "mode": mode,
        "entry_count": len(documents),
        "changed_entry_count": changed_count,
        "embedding_policy_sha256": validated["embedding_policy_sha256"],
    }


def visual_vector_search_results(
    store: Any,
    index: Mapping[str, Any],
    query: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search FAISS and restore source-grounded provenance from the sidecar."""

    validated = validate_visual_vector_index(index)
    query = _query_text(query, allow_empty=True)
    if not query:
        return []
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= _MAX_SEARCH_LIMIT
    ):
        raise ValueError(
            f"visual vector search limit must be between 1 and {_MAX_SEARCH_LIMIT}"
        )
    by_path = {record["artifact_path"]: record for record in validated["records"]}
    results = []
    for hit in store.search(query, top_k=limit, level="l2"):
        record = by_path.get(hit.file)
        if record is None:
            raise ValueError("visual vector store result is absent from its sidecar")
        results.append(
            {
                "artifact_path": record["artifact_path"],
                "score": float(hit.score),
                "caption": record.get("caption", ""),
                "role_hint": record.get("role_hint", ""),
                "mime_type": record.get("mime_type", ""),
                "source_paths": list(record.get("source_paths") or ()),
                "symbols": list(record.get("symbols") or ()),
                "entry_sha256": record["entry_sha256"],
            }
        )
    return results


def _changed_entries(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> tuple[int, set[str]]:
    current_records = {record["artifact_path"]: record for record in current["records"]}
    if previous is None:
        return len(current_records), {
            _content_hash(record["embedding_text"])
            for record in current_records.values()
        }
    previous_records = {
        record["artifact_path"]: record for record in previous["records"]
    }
    if previous["embedding_policy_sha256"] != current["embedding_policy_sha256"]:
        changed_paths = set(previous_records) | set(current_records)
    else:
        changed_paths = {
            path
            for path in set(previous_records) | set(current_records)
            if previous_records.get(path, {}).get("entry_sha256")
            != current_records.get(path, {}).get("entry_sha256")
        }
    hashes = {
        _content_hash(record["embedding_text"])
        for path in changed_paths
        for record in (previous_records.get(path), current_records.get(path))
        if record is not None
    }
    return len(changed_paths), hashes


def _canonicalize_store_rows(store: Any) -> None:
    documents = list(getattr(store, "l2_documents", ()) or ())
    for row, document in enumerate(documents):
        metadata = dict(document.metadata)
        metadata["chunk_id"] = row
        document.metadata = metadata


def _content_hash(text: str) -> str:
    return hashlib.md5(  # noqa: S324 - vector row compatibility
        text.encode("utf-8", errors="replace"),
        usedforsecurity=False,
    ).hexdigest()


def _normalized_vector(vector: Sequence[float], *, dimensions: int) -> list[float]:
    if isinstance(vector, (str, bytes, bytearray)):
        raise ValueError("visual query embedding has invalid dimensions")
    try:
        values = list(vector)
    except TypeError as exc:
        raise ValueError("visual query embedding must be numeric") from exc
    if len(values) != dimensions:
        raise ValueError("visual query embedding has invalid dimensions")
    normalized = []
    for value in values:
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError("visual query embedding contains a non-finite value")
        normalized.append(float(value))
    norm = math.sqrt(sum(value * value for value in normalized))
    if not norm:
        return [0.0] * dimensions
    return [value / norm for value in normalized]


def _validate_threshold(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or value > 1
    ):
        raise ValueError("visual vector delta threshold must be between 0 and 1")


def _query_text(value: Any, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError("visual vector query must be text")
    text = value.strip()
    if not text and not allow_empty:
        raise ValueError("visual vector query must not be empty")
    if len(text.encode("utf-8")) > _MAX_QUERY_BYTES:
        raise ValueError("visual vector query exceeds the byte limit")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in text):
        raise ValueError("visual vector query contains control characters")
    return text


def _index_type(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in {"flat", "ivf"}:
        raise ValueError("visual vector store index type must be 'flat' or 'ivf'")
    return normalized


__all__ = [
    "VISUAL_VECTOR_DOCUMENT_TYPE",
    "VISUAL_VECTOR_STORE_SCHEMA",
    "VisualQueryEmbedding",
    "VisualVectorDocument",
    "create_visual_vector_store",
    "update_visual_vector_store",
    "visual_vector_documents",
    "visual_vector_search_results",
    "visual_vector_store_identity",
]
