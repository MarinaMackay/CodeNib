from __future__ import annotations

from typing import List, Optional

from ..index.embedding import CodeVectorStore, build_hierarchical_vector_store
from ..log_utils import get_logger
from ..types import QueriedNode

logger = get_logger(__name__)


class EmbeddingRetrievePipeline:
    """Embedding-only retrieval pipeline.

    Builds a hierarchical vector store from the repository and retrieves
    top-K nodes using embedding similarity search.

    Args:
        repo_path: Repository root to index.
        index_path: Directory used for vector index caches.
        embedding_model: Embedding model name.
        embedding_provider: Embedding provider (huggingface or openai).
        embedding_dimension: Embedding vector dimension.
        languages: Languages to chunk for indexing (default: ["python"]).
        max_lines_per_chunk: Maximum lines per chunk.
        top_k: Default number of results to retrieve.
    """

    def __init__(
        self,
        repo_path: str,
        index_path: str,
        *,
        embedding_model: str = "nomic-ai/CodeRankEmbed",
        embedding_provider: str = "huggingface",
        embedding_dimension: int = 768,
        languages: Optional[List[str]] = None,
        max_lines_per_chunk: int = 300,
        top_k: int = 50,
    ) -> None:
        self.top_k = top_k
        self.vector_store: CodeVectorStore = build_hierarchical_vector_store(
            repo_path=repo_path,
            index_path=index_path,
            plan_name=None,
            languages=languages or ["python"],
            max_lines_per_chunk=max_lines_per_chunk,
            build_levels=["l2"],
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            embedding_dimension=embedding_dimension,
            embedding_kwargs={
                "model_kwargs": {"trust_remote_code": True},
            },
            index_metric="ip",
        )

    def query(self, query: str, top_k: Optional[int] = None) -> List[QueriedNode]:
        """Retrieve top-K nodes using embedding similarity search.

        Args:
            query: The search query (e.g., problem statement).
            top_k: Number of results; overrides the instance default if provided.

        Returns:
            List of QueriedNode sorted by similarity score.
        """
        k = top_k if top_k is not None else self.top_k
        search_results = self.vector_store.search_with_content(query, top_k=k)
        return [
            QueriedNode(
                node_name=n.node_name,
                type=n.type,
                file=n.file,
                node_id=n.node_id,
                start_line=n.start_line,
                end_line=n.end_line,
                score=n.score,
                content=n.content,
            )
            for n in search_results
        ]

    def close(self) -> None:
        """Release vector store resources."""
        if self.vector_store is not None:
            self.vector_store.close()
            self.vector_store = None
