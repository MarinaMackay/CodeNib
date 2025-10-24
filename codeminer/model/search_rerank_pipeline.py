import os
from pathlib import Path
from typing import List, Optional

from ..agent.rerank_agent import RerankAgent
from ..code_chunker import CodeChunker
from ..embedding import CodeVectorStore
from ..llm.llm_config import LLMConfig, LLMProvider
from ..log_utils import get_logger
from ..types import NodeWithScoreContent

logger = get_logger(__name__)


class SearchRerankPipeline:
    r"""A classical code retrieval baseline using Search + Rerank.

    :class:`~codeminer.model.SearchRerankPipeline` implements a two-stage
    retrieval process for code localization:

    **Stage 1 - Dense Retrieval:** Retrieves top-k code chunks using semantic
    similarity search:

    .. math::
        \mathcal{C}_{\text{top-k}} = \text{TopK}(\text{sim}(\mathbf{e}_q, \mathbf{e}_c) 
        \mid c \in \mathcal{C})

    where :math:`\mathbf{e}_q` is the query embedding, :math:`\mathbf{e}_c` is
    the code chunk embedding, and :math:`\mathcal{C}` is the set of all code
    chunks in the repository.

    **Stage 2 - Reranking:** Refines the ranking using a specialized reranking
    model:

    .. math::
        \mathcal{C}^* = \text{Rerank}(\mathcal{C}_{\text{top-k}}, q)

    where :math:`q` is the query and :math:`\mathcal{C}^*` is the reranked
    results.

    This approach serves as a baseline method in code information retrieval,
    combining the efficiency of vector search with the precision of neural
    reranking.

    Args:
        repo_path (str): Path to the repository to index and search.
        embedding_model (str, optional): Name of the embedding model.
            (default: :obj:`"text-embedding-ada-002"`)
        embedding_provider (str, optional): Embedding model provider.
            (default: :obj:`"openai"`)
        embedding_dimension (int, optional): Dimension of embedding vectors.
            (default: :obj:`1536`)
        embedding_model_kwargs (dict, optional): Additional kwargs for embedding
            model initialization. (default: :obj:`None`)
        rerank_model (str, optional): Name of the reranking model.
            (default: :obj:`"nomic-ai/CodeRankLLM"`)
        rerank_provider (LLMProvider, optional): Reranking model provider.
            (default: :obj:`LLMProvider.VLLM_OPENAI`)
        rerank_temperature (float, optional): Temperature for reranking model.
            (default: :obj:`0.0`)
        rerank_max_tokens (int, optional): Maximum tokens for reranking.
            (default: :obj:`4096`)
        languages (List[str], optional): Programming languages to index.
            (default: :obj:`["python"]`)
        max_lines_per_chunk (int, optional): Maximum lines per code chunk.
            (default: :obj:`100`)
        cache_dir (str, optional): Directory for caching vector store indices.
            If set to :obj:`None`, defaults to :obj:`~/.codeminer`.
            (default: :obj:`None`)
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
        rerank_max_tokens: int = 2048,
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
    
    def query(self, query: str, top_k: int = 10) -> List[NodeWithScoreContent]:
        """
        Query the vector store and rerank the results.
        """
        nodes_with_content = self.vector_store.search_with_content(query=query, top_k=top_k)
        content_map = {node.node_name: node.content for node in nodes_with_content}
        
        ranked_nodes = self.rerank_agent.rerank_nodes(query, nodes_with_content, top_k=top_k)
        
        results = []
        for node in ranked_nodes:
            result = NodeWithScoreContent(
                node_name=node.node_name,
                type=node.type,
                file=node.file,
                start_line=node.start_line,
                end_line=node.end_line,
                score=node.score,
                content=content_map.get(node.node_name),
            )
            results.append(result)

        return results

