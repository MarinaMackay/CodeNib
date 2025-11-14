from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..code_chunker import CodeChunker
from ..index.embedding import CodeVectorStore
from ..index.sparse_idx.bm25_index import BM25CodeIndexer
from ..llm.llm_config import LLMConfig, LLMProvider
from ..log_utils import get_logger
from ..ops.rerank import RerankContext, register_rerank_ops
from ..ops.retrieve import RetrieveContext, register_retrieve_ops
from ..plans.execution import ExecutionEngine
from ..plans.ir_exec import ExecutionGraph, ExecutionNode
from ..plans.ir_physical import PhysicalOperator
from ..types import QueriedNode

logger = get_logger(__name__)

SUPPORTED_ENGINES = {"dense", "sparse"}


@dataclass(frozen=True)
class RetrieveStageConfig:
    """Declarative configuration for an individual retrieval branch."""

    engine: str = "dense"
    weight: float = 1.0
    top_k: Optional[int] = None
    params: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.engine not in SUPPORTED_ENGINES:
            raise ValueError(
                f"Unsupported retrieval engine '{self.engine}'. "
                f"Expected one of: {', '.join(sorted(SUPPORTED_ENGINES))}"
            )
        if self.weight <= 0:
            raise ValueError("Retrieval stage weight must be positive.")


def build_retrieve_plan(mode: str = "dense") -> List[RetrieveStageConfig]:
    """Convenience helper for common retrieval plan presets."""

    normalized = (mode or "dense").strip().lower()
    if normalized == "dense":
        return [RetrieveStageConfig(engine="dense", top_k=50)]
    if normalized == "sparse":
        return [RetrieveStageConfig(engine="sparse", top_k=50)]
    if normalized == "hybrid":
        return [
            RetrieveStageConfig(engine="dense", weight=0.6, top_k=64),
            RetrieveStageConfig(engine="sparse", weight=0.4, top_k=64),
        ]
    raise ValueError(
        f"Unknown retrieval mode '{mode}'. Choose from: dense, sparse, hybrid."
    )


class RetrieveRerankPipeline:
    r"""Composable code retrieval pipeline built from retrieval + rerank ops.

    The pipeline wires :mod:`codeminer.ops.retrieve` (sparse/dense/hybrid) with
    :mod:`codeminer.ops.rerank` to form a two-stage search baseline. Retrieval
    branches are described through :class:`RetrieveStageConfig` objects, which
    can be combined (e.g., hybrid BM25 + embedding) and weighted before
    feeding results to an LLM reranker.

    Args:
        repo_path: Repository root to index.
        index_path: Directory used for vector index caches.
        retrieval_plan: Optional sequence of :class:`RetrieveStageConfig`.
            When omitted, :func:`build_retrieve_plan` is invoked with the
            provided ``retrieval_mode``.
        retrieval_mode: Shortcut for :func:`build_retrieve_plan`.
        embedding_model / provider / dimension / kwargs: Dense index config.
        rerank_model / provider / temperature / max_tokens: Reranker config.
        languages: Languages to chunk for indexing (default: ["python"]).
        max_lines_per_chunk: Maximum lines per chunk passed to chunker.
        sparse_max_k: Upper bound for BM25 index fan-out; defaults to 128.
        rerank_window_size / rerank_window_step: Sliding window controls for the
            reranker (see :meth:`RerankAgent.rerank_nodes`). When ``None``, the
            reranker considers all candidates at once.
    """

    def __init__(
        self,
        repo_path: str,
        index_path: str,
        *,
        retrieval_plan: Optional[Sequence[RetrieveStageConfig]] = None,
        retrieval_mode: str = "dense",
        embedding_model: str = "nomic-ai/CodeRankEmbed",
        embedding_provider: str = "huggingface",
        embedding_dimension: int = 768,
        embedding_model_kwargs: Optional[dict] = None,
        rerank_model: str = "Qwen/Qwen2.5-Coder-7B",
        rerank_provider: LLMProvider = LLMProvider.VLLM_OPENAI,
        rerank_temperature: float = 0.0,
        rerank_max_tokens: int = 2048,
        languages: Optional[List[str]] = None,
        max_lines_per_chunk: int = 100,
        sparse_max_k: int = 128,
        rerank_window_size: Optional[int] = None,
        rerank_window_step: Optional[int] = None,
    ) -> None:
        self.repo_path = self._validate_repo(repo_path)
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.languages = languages or ["python"]
        self.max_lines_per_chunk = max_lines_per_chunk
        self._chunks = None

        plan = (
            list(retrieval_plan)
            if retrieval_plan
            else build_retrieve_plan(retrieval_mode)
        )
        if not plan:
            raise ValueError("Retrieval plan must contain at least one stage.")
        self.retrieve_plan: List[RetrieveStageConfig] = plan
        self._default_stage_top_k = (
            max((stage.top_k or 0) for stage in self.retrieve_plan) or 32
        )

        self.vector_store: Optional[CodeVectorStore] = None
        self.bm25_index: Optional[BM25CodeIndexer] = None

        embedding_kwargs = self._prepare_embedding_kwargs(embedding_model_kwargs)
        if self._needs_engine("dense"):
            self.vector_store = self._initialize_vector_store(
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                embedding_dimension=embedding_dimension,
                embedding_kwargs=embedding_kwargs,
            )

        if self._needs_engine("sparse"):
            sparse_cap = max(
                sparse_max_k,
                max(
                    (stage.top_k or self._default_stage_top_k)
                    for stage in self.retrieve_plan
                    if stage.engine == "sparse"
                ),
            )
            self.bm25_index = self._initialize_bm25_index(max_k=sparse_cap)

        llm_config = LLMConfig(
            model_name=rerank_model,
            provider=rerank_provider,
            max_tokens=rerank_max_tokens,
            temperature=rerank_temperature,
            config_data={"VLLM_TRUST_REMOTE_CODE": "true"},
        )

        self.rerank_window_size = (
            rerank_window_size
            if rerank_window_size and rerank_window_size > 0
            else None
        )
        self.rerank_window_step = (
            rerank_window_step
            if rerank_window_step and rerank_window_step > 0
            else None
        )

        self.engine = ExecutionEngine()
        self.retrieve_context = RetrieveContext(
            bm25=self.bm25_index,
            vector_store=self.vector_store,
            regex_index=None,
            default_top_k=self._default_stage_top_k,
        )
        register_retrieve_ops(self.engine, self.retrieve_context)

        self.rerank_context = RerankContext(
            llm_config=llm_config,
            window_size=rerank_window_size,
            window_step=rerank_window_step,
        )
        register_rerank_ops(self.engine, self.rerank_context)

        logger.info(
            "RetrieveRerankPipeline initialized",
            extra={
                "repo": self.repo_path,
                "index_path": str(self.index_path),
                "retrieval_plan": [stage.engine for stage in self.retrieve_plan],
                "dense_index": bool(self.vector_store),
                "sparse_index": bool(self.bm25_index),
                "rerank_model": rerank_model,
            },
        )

    def query(self, query: str, top_k: int = 10) -> List[QueriedNode]:
        """Execute retrieve + rerank plan for the provided query."""
        graph, rerank_node_id = self._build_execution_graph(query, top_k)
        state = self.engine.execute(graph)
        results = state.get(rerank_node_id, [])
        if not isinstance(results, list):
            return []
        nodes: List[QueriedNode] = []
        for item in results[:top_k]:
            if isinstance(item, QueriedNode):
                nodes.append(item)
        return nodes

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _validate_repo(self, repo_path: str) -> str:
        resolved = os.path.abspath(repo_path)
        if not (os.path.exists(resolved) and os.path.isdir(resolved)):
            raise ValueError(
                f"Repository path is invalid (does not exist or is not a directory): {resolved}"
            )
        return resolved

    def _prepare_embedding_kwargs(
        self, embedding_model_kwargs: Optional[dict]
    ) -> Dict[str, object]:
        if not embedding_model_kwargs:
            return {}
        embedding_kwargs: Dict[str, object] = {}
        if "model_kwargs" in embedding_model_kwargs:
            embedding_kwargs["model_kwargs"] = embedding_model_kwargs["model_kwargs"]
        if "encode_kwargs" in embedding_model_kwargs:
            embedding_kwargs["encode_kwargs"] = embedding_model_kwargs["encode_kwargs"]
        if embedding_model_kwargs.get("trust_remote_code"):
            model_kwargs = embedding_kwargs.setdefault("model_kwargs", {})
            model_kwargs["trust_remote_code"] = True
        return embedding_kwargs

    def _needs_engine(self, engine: str) -> bool:
        return any(stage.engine == engine for stage in self.retrieve_plan)

    def _initialize_vector_store(
        self,
        *,
        embedding_model: str,
        embedding_provider: str,
        embedding_dimension: int,
        embedding_kwargs: Dict[str, object],
    ) -> CodeVectorStore:
        vector_store = CodeVectorStore(
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            dimension=embedding_dimension,
            store_path=str(self.index_path),
            **embedding_kwargs,
        )
        model_suffix = embedding_model.replace("/", "__")
        config_file = self.index_path / f"config_{model_suffix}.json"

        if config_file.exists():
            logger.info(
                "Loading vector store from cache.",
                extra={"index_path": str(self.index_path)},
            )
            vector_store.load(str(self.index_path))
        else:
            logger.info("Building vector store index.")
            chunks = self._ensure_chunks()
            chunks_for_indexing = [chunk._asdict() for chunk in chunks]
            vector_store.add_code_chunks(chunks_for_indexing)
            vector_store.save(str(self.index_path))
            logger.info(
                "Vector store built and cached.",
                extra={"chunk_count": len(chunks_for_indexing)},
            )
        return vector_store

    def _initialize_bm25_index(self, *, max_k: int) -> BM25CodeIndexer:
        logger.info("Building BM25 index.", extra={"max_k": max_k})
        chunks = self._ensure_chunks()
        return BM25CodeIndexer(chunks=chunks, max_k=max_k)

    def _ensure_chunks(self):
        if self._chunks is not None:
            return self._chunks
        primary_language = self.languages[0] if self.languages else "python"
        chunker = CodeChunker(
            language=primary_language,
            max_lines_per_chunk=self.max_lines_per_chunk,
        )
        chunks = chunker.chunk_repository(
            repo_path=self.repo_path,
            languages=self.languages,
        )
        if not chunks:
            raise ValueError("No code chunks generated from repository.")
        self._chunks = chunks
        return self._chunks

    def _build_execution_graph(
        self, query: str, top_k: int
    ) -> Tuple[ExecutionGraph, str]:
        graph = ExecutionGraph()
        retrieve_nodes: List[str] = []
        for idx, stage in enumerate(self.retrieve_plan):
            node_id = f"retrieve_{idx}"
            params = dict(stage.params)
            stage_top_k = stage.top_k or max(top_k, self._default_stage_top_k)
            params.setdefault("query", query)
            params.setdefault("top_k", stage_top_k)
            params.setdefault("return_content", True)
            graph.add_node(
                ExecutionNode(
                    node_id=node_id,
                    operator=self._operator_for_stage(stage),
                    params=params,
                    deps=[],
                )
            )
            retrieve_nodes.append(node_id)

        if len(retrieve_nodes) == 1:
            aggregate_node_id = retrieve_nodes[0]
        else:
            aggregate_node_id = "hybrid_0"
            weights = [stage.weight for stage in self.retrieve_plan]
            graph.add_node(
                ExecutionNode(
                    node_id=aggregate_node_id,
                    operator=PhysicalOperator.HYBRID_RETRIEVE.value,
                    params={
                        "weights": weights,
                        "top_k": max(top_k, self._default_stage_top_k),
                    },
                    deps=retrieve_nodes,
                )
            )

        rerank_node_id = "rerank_0"
        rerank_params = {"query": query, "top_k": top_k, "return_content": True}
        if self.rerank_window_size is not None:
            rerank_params["window_size"] = self.rerank_window_size
        if self.rerank_window_step is not None:
            rerank_params["window_step"] = self.rerank_window_step

        graph.add_node(
            ExecutionNode(
                node_id=rerank_node_id,
                operator=PhysicalOperator.LLM_RERANK.value,
                params=rerank_params,
                deps=[aggregate_node_id],
            )
        )

        return graph, rerank_node_id

    def _operator_for_stage(self, stage: RetrieveStageConfig) -> str:
        if stage.engine == "dense":
            return PhysicalOperator.FAISS_RETRIEVE.value
        if stage.engine == "sparse":
            return PhysicalOperator.BM25_TOPK.value
        raise ValueError(f"Unsupported engine '{stage.engine}'.")
