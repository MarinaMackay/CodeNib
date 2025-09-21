"""
Vector Store implementation using FAISS and LlamaIndex for code embeddings.
This module provides functionality to create, store, and query vector embeddings
of code chunks for semantic similarity search.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS is required but not installed. Install with: pip install faiss-cpu"
    )

from ..log_utils import get_logger
from ..types import NodeWithContent, NodeWithScore

logger = get_logger(__name__)


class CodeVectorStore:
    """
    Vector store for code embeddings using FAISS and LlamaIndex.
    Provides semantic search capabilities over code chunks.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-ada-002",
        embedding_provider: str = "openai",
        dimension: int = 1536,
        index_type: str = "flat",
        store_path: Optional[str] = None,
        **embedding_kwargs,
    ):
        """
        Initialize the CodeVectorStore.

        Args:
            embedding_model: Name of the embedding model to use
            embedding_provider: Provider for embeddings ("openai", "huggingface")
            dimension: Dimension of the embedding vectors
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
            store_path: Path to store/load the vector store
            **embedding_kwargs: Additional arguments for embedding model
        """
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.dimension = dimension
        self.index_type = index_type
        self.store_path = Path(store_path) if store_path else None

        # Initialize embedding model
        self.embedding = self._initialize_embedding_model(**embedding_kwargs)
        Settings.embed_model = self.embedding
        # Disable LLM when not using OpenAI to avoid API key requirement
        if self.embedding_provider.lower() != "openai":
            Settings.llm = None

        # Initialize FAISS index
        self.faiss_index = self._create_faiss_index()
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)

        # Initialize storage context and index
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.index = None
        self.retriever = None
        self.query_engine = None

        # Metadata storage
        self.node_metadata = {}
        self.nodes = []

        logger.info(
            f"Initialized CodeVectorStore with {embedding_provider}:{embedding_model}"
        )

    def _initialize_embedding_model(self, **kwargs) -> BaseEmbedding:
        """Initialize the embedding model based on provider."""
        if self.embedding_provider.lower() == "openai":
            return OpenAIEmbedding(model=self.embedding_model, **kwargs)
        elif self.embedding_provider.lower() == "huggingface":
            # Default to a good code embedding model if not specified
            model_name = self.embedding_model
            if model_name == "text-embedding-ada-002":  # Default OpenAI model
                model_name = "microsoft/codebert-base"

            return HuggingFaceEmbedding(model_name=model_name, **kwargs)
        else:
            raise ValueError(
                f"Unsupported embedding provider: {self.embedding_provider}"
            )

    def _create_faiss_index(self):
        """Create FAISS index based on type and dimension."""
        if self.index_type.lower() == "flat":
            return faiss.IndexFlatL2(self.dimension)
        elif self.index_type.lower() == "ivf":
            # IVF index with 100 clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type.lower() == "hnsw":
            # HNSW index for fast approximate search
            return faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

    def add_code_chunks(self, code_chunks: List[Dict[str, Any]]) -> None:
        """
        Add code chunks to the vector store.

        Args:
            code_chunks: List of code chunk dictionaries with content and metadata
        """
        if not code_chunks:
            logger.warning("No code chunks provided")
            return

        logger.info(f"Adding {len(code_chunks)} code chunks to vector store")

        # Convert chunks to TextNode objects
        nodes = []
        for i, chunk in enumerate(code_chunks):
            # Extract content and metadata
            content = chunk.get("content", "")
            metadata = {
                "chunk_id": i,
                "chunk_type": chunk.get("chunk_type", "unknown"),
                "name": chunk.get("name", f"chunk_{i}"),
                "file": chunk.get("file", ""),
                "start_line": chunk.get("start_line", 0),
                "end_line": chunk.get("end_line", 0),
            }

            # Add any additional metadata
            for key, value in chunk.items():
                if key not in ["content"] and key not in metadata:
                    metadata[key] = value

            # Create TextNode
            node = TextNode(text=content, id_=f"chunk_{i}", metadata=metadata)
            nodes.append(node)

            # Store metadata for quick access
            self.node_metadata[node.id_] = metadata

        # Store nodes
        self.nodes.extend(nodes)

        # Create or update index
        if self.index is None:
            self.index = VectorStoreIndex(nodes, storage_context=self.storage_context)
        else:
            # Add nodes to existing index
            for node in nodes:
                self.index.insert(node)

        # Update retriever and query engine
        self._update_retriever()

        logger.info(f"Successfully added {len(nodes)} nodes to vector store")

    def add_nodes_with_content(self, nodes: List[NodeWithContent]) -> None:
        """
        Add NodeWithContent objects to the vector store.

        Args:
            nodes: List of NodeWithContent objects
        """
        chunks = []
        for node in nodes:
            chunk = {
                "content": node.content,
                "chunk_type": node.type,
                "name": node.node_name,
                "file": node.file,
                "start_line": node.start_line,
                "end_line": node.end_line,
            }
            chunks.append(chunk)

        self.add_code_chunks(chunks)

    def _update_retriever(self):
        """Update the retriever and query engine after adding nodes."""
        if self.index is not None:
            self.retriever = VectorIndexRetriever(index=self.index, similarity_top_k=10)
            self.query_engine = RetrieverQueryEngine(retriever=self.retriever)

    def search(
        self, query: str, top_k: int = 10, score_threshold: Optional[float] = None
    ) -> List[NodeWithScore]:
        """
        Search for similar code chunks using semantic similarity.

        Args:
            query: Search query text
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of NodeWithScore objects
        """
        if self.retriever is None:
            logger.warning("No retriever available. Add code chunks first.")
            return []

        logger.debug(f"Searching for: {query[:100]}...")

        # Update retriever top_k
        self.retriever.similarity_top_k = top_k

        # Perform search
        retrieved_nodes = self.retriever.retrieve(query)

        # Convert to NodeWithScore objects
        results = []
        for node in retrieved_nodes:
            metadata = node.node.metadata
            score = float(node.score) if node.score is not None else 0.0

            # Apply score threshold if specified
            if score_threshold is not None and score < score_threshold:
                continue

            node_with_score = NodeWithScore(
                node_name=metadata.get("name", "unknown"),
                type=metadata.get("chunk_type", "unknown"),
                file=metadata.get("file", ""),
                start_line=metadata.get("start_line", 0),
                end_line=metadata.get("end_line", 0),
                score=score,
            )
            results.append(node_with_score)

        logger.debug(f"Found {len(results)} results")
        return results

    def search_with_content(
        self, query: str, top_k: int = 10, score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search and return results with content included.

        Args:
            query: Search query text
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of dictionaries with node information and content
        """
        if self.retriever is None:
            logger.warning("No retriever available. Add code chunks first.")
            return []

        # Update retriever top_k
        self.retriever.similarity_top_k = top_k

        # Perform search
        retrieved_nodes = self.retriever.retrieve(query)

        # Convert to detailed results
        results = []
        for node in retrieved_nodes:
            metadata = node.node.metadata
            score = float(node.score) if node.score is not None else 0.0

            # Apply score threshold if specified
            if score_threshold is not None and score < score_threshold:
                continue

            result = {
                "score": score,
                "content": node.node.text,
                "name": metadata.get("name", "unknown"),
                "chunk_type": metadata.get("chunk_type", "unknown"),
                "file": metadata.get("file", ""),
                "start_line": metadata.get("start_line", 0),
                "end_line": metadata.get("end_line", 0),
                "metadata": metadata,
            }
            results.append(result)

        return results

    def query_with_context(self, query: str) -> str:
        """
        Query with context using the query engine.

        Args:
            query: Natural language query

        Returns:
            Generated response with context
        """
        if self.query_engine is None:
            return "No query engine available. Add code chunks first."

        response = self.query_engine.query(query)
        return str(response)

    def save(self, path: Optional[str] = None) -> None:
        """
        Save the vector store to disk.

        Args:
            path: Path to save the store (uses self.store_path if not provided)
        """
        save_path = Path(path) if path else self.store_path
        if save_path is None:
            raise ValueError("No save path provided")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving vector store to {save_path}")

        # Save FAISS index
        faiss_path = save_path / "faiss_index.bin"
        faiss.write_index(self.faiss_index, str(faiss_path))

        # Save metadata
        metadata_path = save_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.node_metadata, f, indent=2)

        # Save nodes
        nodes_path = save_path / "nodes.pkl"
        with open(nodes_path, "wb") as f:
            pickle.dump(self.nodes, f)

        # Save configuration
        config_path = save_path / "config.json"
        config = {
            "embedding_model": self.embedding_model,
            "embedding_provider": self.embedding_provider,
            "dimension": self.dimension,
            "index_type": self.index_type,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Vector store saved successfully")

    def load(self, path: Optional[str] = None) -> None:
        """
        Load the vector store from disk.

        Args:
            path: Path to load the store from (uses self.store_path if not provided)
        """
        load_path = Path(path) if path else self.store_path
        if load_path is None:
            raise ValueError("No load path provided")

        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store not found at {load_path}")

        logger.info(f"Loading vector store from {load_path}")

        # Load configuration
        config_path = load_path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)

            # Verify configuration matches
            if config.get("dimension") != self.dimension:
                logger.warning(
                    f"Dimension mismatch: expected {self.dimension}, "
                    f"got {config.get('dimension')}"
                )

        # Load FAISS index
        faiss_path = load_path / "faiss_index.bin"
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
            self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

        # Load metadata
        metadata_path = load_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.node_metadata = json.load(f)

        # Load nodes
        nodes_path = load_path / "nodes.pkl"
        if nodes_path.exists():
            with open(nodes_path, "rb") as f:
                self.nodes = pickle.load(f)

        # Recreate index if we have nodes
        if self.nodes:
            self.index = VectorStoreIndex(
                self.nodes, storage_context=self.storage_context
            )
            self._update_retriever()

        logger.info(f"Vector store loaded successfully with {len(self.nodes)} nodes")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with store statistics
        """
        stats = {
            "total_nodes": len(self.nodes),
            "embedding_model": self.embedding_model,
            "embedding_provider": self.embedding_provider,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "faiss_index_size": self.faiss_index.ntotal if self.faiss_index else 0,
        }

        if self.nodes:
            # Analyze chunk types
            chunk_types = {}
            for node in self.nodes:
                chunk_type = node.metadata.get("chunk_type", "unknown")
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            stats["chunk_types"] = chunk_types

        return stats

    def clear(self) -> None:
        """Clear all data from the vector store."""
        logger.info("Clearing vector store")

        # Reset FAISS index
        self.faiss_index = self._create_faiss_index()
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        # Clear data
        self.index = None
        self.retriever = None
        self.query_engine = None
        self.node_metadata = {}
        self.nodes = []

        logger.info("Vector store cleared")


def create_code_vector_store(
    embedding_model: str = "text-embedding-ada-002",
    embedding_provider: str = "openai",
    store_path: Optional[str] = None,
    **kwargs,
) -> CodeVectorStore:
    """
    Factory function to create a CodeVectorStore.

    Args:
        embedding_model: Name of the embedding model
        embedding_provider: Provider for embeddings
        store_path: Path to store/load the vector store
        **kwargs: Additional arguments for CodeVectorStore

    Returns:
        CodeVectorStore instance
    """
    return CodeVectorStore(
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        store_path=store_path,
        **kwargs,
    )
