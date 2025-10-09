import os
from pathlib import Path
from typing import List, Optional

from codeminer.code_chunker import CodeChunker
from codeminer.embedding import CodeVectorStore
from codeminer.log_utils import get_logger

logger = get_logger(__name__)


class SearchRerankPipeline:
    """
    Pipeline for Search + Rerank Pipeline.

    Workflow:
    1. Initialize with embedding and rerank model configs
    #TODO: Add rerank
    #TODO: Add query
    """

    def __init__(
        self,
        # Repo config
        repo_path: str,
        # Embedding config
        embedding_model: str = "text-embedding-ada-002",
        embedding_provider: str = "openai",
        embedding_dimension: int = 1536,
        #TODO: Rerank config
        rerank_model: str = "Salesforce/SweRankLLM-Small",
        rerank_provider: str = "vllm",  # for local models
        rerank_temperature: float = 0.0,
        # Repo processing config
        languages: Optional[List[str]] = ['python'],
        max_lines_per_chunk: int = 100,
        # Cache config
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize pipeline and build vector database.
        """
        # Attributes
        self.repo_path = None
        self.vector_store = None
        
        # Validate repo_path
        self.repo_path = os.path.abspath(repo_path)
        if not os.path.exists(self.repo_path):
            raise ValueError(f"Repository path does not exist: {self.repo_path}")
        if not os.path.isdir(self.repo_path):
            raise ValueError(f"Repository path is not a directory: {self.repo_path}")
        
        # Initialize index directory
        cache_dir = Path(cache_dir or Path.home() / ".codeminer")
        cache_dir.mkdir(parents=True, exist_ok=True)
        index_path = cache_dir / f"index_{self.repo_path}"
        logger.info(
            f"Initializing Pipeline: "
            f"embed={embedding_provider}:{embedding_model}, "
            f"repo={self.repo_path}"
            f"index_dir={index_path}"
        )
        
        # Chunk repository
        code_chunker = CodeChunker(
            language=languages[0],
            max_lines_per_chunk=max_lines_per_chunk,
        )
        chunks = code_chunker.chunk_repository(
            repo_path=self.repo_path,
            languages=languages,
        )
        logger.info(f"Generated {len(chunks)} chunks")
        if not chunks:
            raise ValueError("No code chunks generated from repository")

        self.vector_store = CodeVectorStore(
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            dimension=embedding_dimension,
        )

        # Build vector index
        chunks_for_indexing = [chunk._asdict() for chunk in chunks]
        self.vector_store.add_code_chunks(chunks_for_indexing)
        self.vector_store.save(str(index_path))
