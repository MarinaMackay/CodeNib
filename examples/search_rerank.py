import os
from pathlib import Path
from typing import List, Optional

from codeminer.agent.rerank_agent import RerankAgent
from codeminer.code_chunker import CodeChunker
from codeminer.embedding import CodeVectorStore
from codeminer.llm.llm_config import LLMConfig, LLMProvider
from codeminer.log_utils import get_logger

logger = get_logger(__name__)


class SearchRerankPipeline:
    """
    Pipeline for Search + Rerank Pipeline.

    Workflow:
    1. Initialize with embedding and rerank model configs
    #TODO: Add query and search methods
    """

    def __init__(
        self,
        # Repo config
        repo_path: str,
        # Embedding config
        embedding_model: str = "text-embedding-ada-002",
        embedding_provider: str = "openai",
        embedding_dimension: int = 1536,
        # Rerank config
        rerank_model: str = "nomic-ai/CodeRankLLM",
        rerank_provider: LLMProvider = LLMProvider.VLLM_OPENAI,
        rerank_temperature: float = 0.0,
        rerank_max_tokens: int = 4096,
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
        self.rerank_agent = None
        
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
        
        # Chunk repository
        code_chunker = CodeChunker(
            language=languages[0],
            max_lines_per_chunk=max_lines_per_chunk,
        )
        chunks = code_chunker.chunk_repository(
            repo_path=self.repo_path,
            languages=languages,
        )
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
        
        # Create LLM config for rerank model
        llm_config = LLMConfig(
            model_name=rerank_model,
            provider=rerank_provider,
            max_tokens=rerank_max_tokens,
            temperature=rerank_temperature,
            config_data={
                "VLLM_TRUST_REMOTE_CODE": "true"
            }
        )
        
        # Initialize rerank agent
        self.rerank_agent = RerankAgent(llm_config=llm_config)

        logger.info(
            f"Initializing Pipeline: "
            f"embed={embedding_provider}:{embedding_model}, "
            f"repo={self.repo_path}"
            f"index_dir={index_path} with {len(chunks)} chunks"
            f"rerank={rerank_provider.value}:{rerank_model}"
        )
