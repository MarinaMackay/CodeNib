"""
Index builder protocol, registry, and concrete implementations.

The compiler can invoke builders to create or update indexes
when ``ResourcePlan`` indicates they are missing or stale.

Concrete builders wrap existing index infrastructure:
  - ``BM25IndexBuilder``   → ``BM25CodeIndexer``
  - ``VectorIndexBuilder`` → ``build_hierarchical_vector_store``
  - ``SymbolGraphBuilder`` → ``SCIPPythonIndexer``
  / ``SCIPRustIndexer`` / ``SCIPTypeScriptIndexer``
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .resources import IndexState, IndexStatus

logger = logging.getLogger(__name__)


@runtime_checkable
class IndexBuilder(Protocol):
    """Protocol for index build tools the compiler can invoke."""

    def build(self, scope: str, **kwargs: Any) -> IndexStatus: ...
    def incremental_update(self, scope: str, **kwargs: Any) -> IndexStatus: ...


class IndexBuilderRegistry:
    """Maps index_type names to their builder implementations."""

    def __init__(self) -> None:
        self._builders: Dict[str, IndexBuilder] = {}

    def register(self, index_type: str, builder: IndexBuilder) -> None:
        self._builders[index_type] = builder

    def get(self, index_type: str) -> Optional[IndexBuilder]:
        return self._builders.get(index_type)

    def has(self, index_type: str) -> bool:
        return index_type in self._builders


# ---------------------------------------------------------------------------
# Concrete builders
# ---------------------------------------------------------------------------


@dataclass
class BM25IndexBuilder:
    """Build a BM25 sparse index by wrapping ``BM25CodeIndexer``."""

    languages: List[str] = field(default_factory=lambda: ["python"])
    max_k: int = 128
    max_lines_per_chunk: int = 300

    def build(self, scope: str, **kwargs: Any) -> IndexStatus:
        repo_path: str = kwargs["repo_path"]
        output_dir: str = kwargs["output_dir"]

        from ..code_chunker import CodeChunker, RepoChunkingConfig
        from ..index.sparse_idx.bm25_index import BM25CodeIndexer

        primary = self.languages[0] if self.languages else "python"
        chunker = CodeChunker(
            language=primary,
            repo_config=RepoChunkingConfig(languages=self.languages),
            max_lines_per_chunk=self.max_lines_per_chunk,
        )
        chunks = chunker.chunk_repository(repo_path=repo_path)

        indexer = BM25CodeIndexer(chunks=chunks, max_k=self.max_k)
        os.makedirs(output_dir, exist_ok=True)
        indexer.save_index(output_dir)

        return IndexStatus(
            index_type="bm25",
            state=IndexState.FRESH,
            last_built=time.time(),
            age_seconds=0.0,
            scope=scope,
            path=output_dir,
            metadata={
                "file_count": len(chunks),
                "max_k": self.max_k,
                "languages": list(self.languages),
            },
        )

    def incremental_update(self, scope: str, **kwargs: Any) -> IndexStatus:
        return self.build(scope, **kwargs)


@dataclass
class VectorIndexBuilder:
    """Build a hierarchical embedding index (L0/L2)."""

    languages: List[str] = field(default_factory=lambda: ["python"])
    embedding_model: str = "nomic-ai/CodeRankEmbed"
    embedding_provider: str = "huggingface"
    embedding_dimension: int = 768
    embedding_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"model_kwargs": {"trust_remote_code": True}},
    )
    build_levels: List[str] = field(default_factory=lambda: ["l0", "l2"])
    max_lines_per_chunk: int = 300
    index_metric: str = "ip"

    def build(self, scope: str, **kwargs: Any) -> IndexStatus:
        repo_path: str = kwargs["repo_path"]
        output_dir: str = kwargs["output_dir"]

        from ..index.embedding.builders import build_hierarchical_vector_store

        os.makedirs(output_dir, exist_ok=True)
        vs = build_hierarchical_vector_store(
            repo_path=repo_path,
            index_path=output_dir,
            plan_name=None,
            languages=self.languages,
            max_lines_per_chunk=self.max_lines_per_chunk,
            build_levels=self.build_levels,
            embedding_model=self.embedding_model,
            embedding_provider=self.embedding_provider,
            embedding_dimension=self.embedding_dimension,
            embedding_kwargs=self.embedding_kwargs,
            index_metric=self.index_metric,
        )

        doc_count = {}
        if hasattr(vs, "l0_documents") and vs.l0_documents:
            doc_count["l0"] = len(vs.l0_documents)
        if hasattr(vs, "l2_documents") and vs.l2_documents:
            doc_count["l2"] = len(vs.l2_documents)

        return IndexStatus(
            index_type="vector",
            state=IndexState.FRESH,
            last_built=time.time(),
            age_seconds=0.0,
            scope=scope,
            path=output_dir,
            metadata={
                "embedding_model": self.embedding_model,
                "levels": list(self.build_levels),
                "document_count": doc_count,
            },
        )

    def incremental_update(self, scope: str, **kwargs: Any) -> IndexStatus:
        return self.build(scope, **kwargs)


@dataclass
class SymbolGraphBuilder:
    """Build a SCIP-based symbol graph."""

    language: str = "python"

    def build(self, scope: str, **kwargs: Any) -> IndexStatus:
        repo_path: str = kwargs["repo_path"]
        output_dir: str = kwargs["output_dir"]

        from ..scip_interface import (
            SCIPPythonIndexer,
            SCIPRustIndexer,
            SCIPTypeScriptIndexer,
        )

        _INDEXER_MAP = {
            "python": SCIPPythonIndexer,
            "rust": SCIPRustIndexer,
            "typescript": SCIPTypeScriptIndexer,
            "javascript": SCIPTypeScriptIndexer,
        }

        indexer_cls = _INDEXER_MAP.get(self.language)
        if indexer_cls is None:
            raise ValueError(f"Unsupported language for symbol graph: {self.language}")

        os.makedirs(output_dir, exist_ok=True)
        indexer = indexer_cls(
            project_root=repo_path,
            output_dir=output_dir,
        )
        graph = indexer.run_pipeline(
            output_file=os.path.join(output_dir, "graph.pkl"),
            skip_level=None,
        )

        node_count = 0
        if graph is not None and hasattr(graph, "graph"):
            node_count = len(graph.graph.vs)

        return IndexStatus(
            index_type="symbol_graph",
            state=IndexState.FRESH,
            last_built=time.time(),
            age_seconds=0.0,
            scope=scope,
            path=output_dir,
            metadata={
                "node_count": node_count,
                "language": self.language,
            },
        )

    def incremental_update(self, scope: str, **kwargs: Any) -> IndexStatus:
        return self.build(scope, **kwargs)


# ---------------------------------------------------------------------------
# Convenience registration
# ---------------------------------------------------------------------------


def register_default_builders(
    registry: IndexBuilderRegistry,
    *,
    languages: Optional[List[str]] = None,
    embedding_model: str = "nomic-ai/CodeRankEmbed",
    embedding_dimension: int = 768,
) -> None:
    """Register all standard index builders with sensible defaults."""
    langs = languages or ["python"]
    registry.register("bm25", BM25IndexBuilder(languages=langs))
    registry.register(
        "vector",
        VectorIndexBuilder(
            languages=langs,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
        ),
    )
    registry.register("symbol_graph", SymbolGraphBuilder(language=langs[0]))
