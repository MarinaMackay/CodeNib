import os
from pathlib import Path
from typing import List, Optional

from codeminer.agent.rerank_agent import RerankAgent
from codeminer.code_chunker import CodeChunker
from codeminer.embedding import CodeVectorStore
from codeminer.llm.llm_config import LLMConfig, LLMProvider
from codeminer.log_utils import get_logger
from codeminer.types import NodeWithScore

logger = get_logger(__name__)


class SearchRerankPipeline:
    """
    Pipeline for Search + Rerank Pipeline.

    1. __init__: Initialize with embedding and rerank model configs
    2. query: Query the vector store and rerank the results
    """

    def __init__(
        self,
        # Repo config
        repo_path: str,
        # Embedding config
        embedding_model: str = "text-embedding-ada-002",
        embedding_provider: str = "openai",
        embedding_dimension: int = 1536,
        embedding_model_kwargs: Optional[dict] = None,
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
        
        # Create the index path based on repo and embedding model
        repo_name = os.path.basename(self.repo_path)
        index_path = cache_dir / f"index_{repo_name}_{embedding_model.replace('/', '_')}"
        
        # Prepare embedding kwargs
        embedding_kwargs = {}
        if embedding_model_kwargs:
            # Extract specific kwargs for different purposes
            if "model_kwargs" in embedding_model_kwargs:
                embedding_kwargs["model_kwargs"] = embedding_model_kwargs["model_kwargs"]
            if "encode_kwargs" in embedding_model_kwargs:
                embedding_kwargs["encode_kwargs"] = embedding_model_kwargs["encode_kwargs"]
            if "trust_remote_code" in embedding_model_kwargs:
                if "model_kwargs" not in embedding_kwargs:
                    embedding_kwargs["model_kwargs"] = {}
                embedding_kwargs["model_kwargs"]["trust_remote_code"] = embedding_model_kwargs["trust_remote_code"]

        self.vector_store = CodeVectorStore(
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            dimension=embedding_dimension,
            store_path=str(index_path),
            **embedding_kwargs,
        )

        # Check if cache exists
        cache_exists = (index_path / "config.json").exists()
        
        if cache_exists:
            logger.info(f"Loading existing vector store from cache: {index_path}")
            try:
                self.vector_store.load(str(index_path))
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Rebuilding index...")
                cache_exists = False
        
        if not cache_exists:
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

            chunks_for_indexing = [chunk._asdict() for chunk in chunks]
            self.vector_store.add_code_chunks(chunks_for_indexing)
            self.vector_store.save(str(index_path))
            logger.info(f"Built and saved vector store to cache: {index_path}")
        
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
            f"Pipeline initialized: "
            f"embed={embedding_provider}:{embedding_model}, "
            f"repo={self.repo_path}, "
            f"index_dir={index_path} with {len(self.vector_store.documents)} chunks, "
            f"rerank={rerank_provider.value}:{rerank_model}"
        )
    
    def query(self, query: str, top_k: int = 10) -> List[NodeWithScore]:
        """
        Query the vector store and rerank the results.
        """
        # Search the vector store
        nodes = self.vector_store.search_with_content(query=query, top_k=top_k)
        # Rerank the nodes
        ranked_results = self.rerank_agent.rerank_nodes(query, nodes, top_k=top_k)

        return ranked_results
