"""
Vector Store implementation using FAISS and LlamaIndex for code embeddings.
This module provides functionality to create, store, and query vector embeddings
of code chunks for semantic similarity search.
"""

import json
import pickle
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from ...log_utils import get_logger
from ...profiler import Profiler
from ...types import NodeInfo

logger = get_logger(__name__)


class CodeVectorStore:
    """
    Vector store for code embeddings using FAISS and LangChain.
    Provides semantic search capabilities over code chunks.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-ada-002",
        embedding_provider: str = "openai",
        dimension: int = 1536,
        index_type: str = "flat",
        index_metric: str = "ip",
        store_path: Optional[str] = None,
        index_params: Optional[Dict[str, Any]] = None,
        profiler: Optional[Profiler] = None,
        **embedding_kwargs,
    ):
        """
        Initialize the CodeVectorStore.

        Args:
            embedding_model: Name of the embedding model to use
            embedding_provider: Provider for embeddings ("openai", "huggingface")
            dimension: Dimension of the embedding vectors
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
            index_metric: Distance metric ("ip" for inner product, "l2" for L2 distance)
            store_path: Path to store/load the vector store
            index_params: Additional parameters for index construction (e.g., nlist)
            profiler: Optional profiler instance to capture detailed timings
            **embedding_kwargs: Additional arguments for embedding model
        """
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.dimension = dimension
        self.index_type = index_type
        self.index_metric = index_metric.lower()
        if self.index_metric not in ["ip", "l2"]:
            raise ValueError(
                f"Unsupported index_metric: {index_metric}. Must be 'ip' or 'l2'."
            )
        self.store_path = Path(store_path) if store_path else None
        self.index_params = index_params or {}
        self.profiler = profiler

        # Initialize embedding model
        self.embedding = self._initialize_embedding_model(**embedding_kwargs)

        self.dimension = self._infer_embedding_dimension(dimension)

        # Initialize vector store (will be created when documents are added)
        self.index = self._build_faiss_index()
        self.vector_store = FAISS(
            embedding_function=self.embedding,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.documents = []
        self._index_requires_training = False
        self._min_train_docs = int(self.index_params.get("train_size", 0))

        logger.info(
            f"Initialized CodeVectorStore with {embedding_provider}:{embedding_model}"
        )

    def _initialize_embedding_model(self, **kwargs) -> Embeddings:
        """Initialize the embedding model based on provider."""
        if self.embedding_provider.lower() == "openai":
            return OpenAIEmbeddings(model=self.embedding_model, **kwargs)
        elif self.embedding_provider.lower() == "huggingface":
            # Default to a good code embedding model if not specified
            model_name = self.embedding_model
            model_kwargs = kwargs.pop("model_kwargs", {})
            encode_kwargs = kwargs.pop("encode_kwargs", {})

            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported embedding provider: {self.embedding_provider}"
            )

    def _infer_embedding_dimension(self, expected: Optional[int]) -> int:
        """Probe the embedding model to determine vector dimensionality."""
        probe_text = "codeminer-dimension-probe"
        vector = self.embedding.embed_query(probe_text)
        if not vector:
            raise ValueError("Failed to infer embedding dimension from model output")

        actual_dim = len(vector)
        if expected is not None and actual_dim != expected:
            logger.warning(
                "Embedding dimension mismatch: expected %s, got %s",
                expected,
                actual_dim,
            )
        return actual_dim

    def _profile_section(self, label: str, metadata: Optional[Dict[str, Any]] = None):
        """Return an active profiler section context if profiling is enabled."""
        if self.profiler is None:
            return nullcontext()
        return self.profiler.section(label, metadata)

    def _build_faiss_index(self) -> faiss.Index:
        """Create a FAISS index according to the configured index_type."""
        index_type = self.index_type.lower()
        metric = self.index_metric
        params = self.index_params

        if index_type == "flat":
            if metric == "ip":
                return faiss.IndexFlatIP(self.dimension)
            elif metric == "l2":
                return faiss.IndexFlatL2(self.dimension)
            else:
                raise ValueError(f"Unsupported metric for flat index: {metric}")

        if index_type == "ivf":
            nlist = int(params.get("nlist", 1024))
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(
                quantizer, self.dimension, nlist, faiss.METRIC_L2
            )
            return index

        if index_type == "hnsw":
            m = int(params.get("M", params.get("m", 32)))
            index = faiss.IndexHNSWFlat(self.dimension, m)
            ef_construction = params.get("ef_construction")
            if ef_construction is not None:
                try:
                    faiss.ParameterSpace().set_index_parameter(
                        index, "efConstruction", int(ef_construction)
                    )
                except Exception:
                    logger.warning("Failed to set efConstruction on HNSW index")
            return index

        raise ValueError(f"Unsupported FAISS index type: {self.index_type}")

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

        # Convert chunks to Document objects
        documents = []
        for i, chunk in enumerate(code_chunks):
            # Extract content and metadata
            content = chunk.get("content", "")
            metadata = {
                "chunk_id": len(self.documents) + i,
                "chunk_type": chunk.get("chunk_type", "unknown"),
                "name": chunk.get("name", f"chunk_{i}"),
                "file": chunk.get("file", ""),
                "start_line": chunk.get("start_line", 0),
                "end_line": chunk.get("end_line", 0),
                "node_id": chunk.get("node_id", ""),
            }

            # Add any additional metadata
            for key, value in chunk.items():
                if key not in ["content"] and key not in metadata:
                    metadata[key] = value

            # Create Document
            document = Document(page_content=content, metadata=metadata)
            documents.append(document)

        # Store documents
        self.documents.extend(documents)
        # add profiling section
        with self._profile_section(
            "vector_store_add_documents",
            {"num_documents": len(documents)},
        ):
            self.vector_store.add_documents(
                documents
            )  # this part will majorly blocked by embedding time

        logger.info(f"Successfully added {len(documents)} documents to vector store")

    def add_nodes_with_content(self, nodes: List[NodeInfo]) -> None:
        """
        Add NodeInfo objects (with content) to the vector store.

        Args:
            nodes: List of NodeInfo objects
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

    def search(
        self, query: str, top_k: int = 10, score_threshold: Optional[float] = None
    ) -> List[NodeInfo]:
        """
        Search for similar code chunks using semantic similarity.

        Args:
            query: Search query text
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of NodeInfo objects with scores populated
        """
        if self.vector_store is None:
            logger.warning("No vector store available. Add code chunks first.")
            return []

        logger.debug(f"Searching for: {query[:100]}...")

        # Perform similarity search with scores
        docs_with_scores = self.vector_store.similarity_search_with_score(
            query, k=top_k
        )

        # Convert to NodeInfo objects
        results = []
        for doc, score in docs_with_scores:
            metadata = doc.metadata

            # Apply score threshold if specified (note: FAISS returns distance, lower is better)
            if score_threshold is not None and score > score_threshold:
                continue

            node_with_score = NodeInfo(
                node_name=metadata.get("name", "unknown"),
                type=metadata.get("chunk_type", "unknown"),
                file=metadata.get("file", ""),
                node_id=metadata.get("node_id", ""),
                start_line=metadata.get("start_line", 0),
                end_line=metadata.get("end_line", 0),
                score=float(score),
            )
            results.append(node_with_score)

        logger.debug(f"Found {len(results)} results")
        return results

    def search_with_content(
        self, query: str, top_k: int = 10, score_threshold: Optional[float] = None
    ) -> List[NodeInfo]:
        """
        Search and return results with content included.

        Args:
            query: Search query text
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of NodeInfo objects with content populated
        """
        if self.vector_store is None:
            logger.warning("No vector store available. Add code chunks first.")
            return []

        # Perform similarity search with scores
        docs_with_scores = self.vector_store.similarity_search_with_score(
            query, k=top_k
        )

        # Convert to NodeInfo objects
        results = []
        for doc, score in docs_with_scores:
            metadata = doc.metadata

            # Apply score threshold if specified (note: FAISS returns distance, lower is better)
            if score_threshold is not None and score > score_threshold:
                continue

            node_with_content = NodeInfo(
                node_name=metadata.get("name", "unknown"),
                type=metadata.get("chunk_type", "unknown"),
                file=metadata.get("file", ""),
                node_id=metadata.get("node_id", ""),
                start_line=metadata.get("start_line", 0),
                end_line=metadata.get("end_line", 0),
                score=float(score),
                content=doc.page_content,
            )
            results.append(node_with_content)

        return results

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

        model_suffix = self.embedding_model.replace("/", "__")
        # Save FAISS vector store (LangChain format)
        if self.vector_store is not None:
            index_name = f"index_{model_suffix}"
            self.vector_store.save_local(str(save_path), index_name=index_name)

        # Save documents
        docs_path = save_path / f"documents_{model_suffix}.pkl"
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

        # Save configuration
        config_path = save_path / f"config_{model_suffix}.json"
        config = {
            "embedding_model": self.embedding_model,
            "embedding_provider": self.embedding_provider,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "index_metric": self.index_metric,
            "index_params": self.index_params,
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
        self._index_requires_training = False

        model_suffix = self.embedding_model.replace("/", "__")
        # Load configuration
        config_path = load_path / f"config_{model_suffix}.json"
        # Fallback to old format if model-specific config doesn't exist
        if not config_path.exists():
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
            saved_index_type = config.get("index_type")
            if saved_index_type and saved_index_type != self.index_type:
                logger.warning(
                    "Index type mismatch: expected %s, got %s",
                    self.index_type,
                    saved_index_type,
                )
            saved_metric = config.get("index_metric")
            if saved_metric and saved_metric != self.index_metric:
                logger.warning(
                    "Index metric mismatch: expected %s, got %s",
                    self.index_metric,
                    saved_metric,
                )
                self.index_metric = saved_metric
            saved_params = config.get("index_params")
            if saved_params and saved_params != self.index_params:
                logger.warning("Index params mismatch between config and runtime")
                self.index_params = saved_params

        # Load FAISS vector store (LangChain format)
        try:
            index_name = f"index_{model_suffix}"
            self.vector_store = FAISS.load_local(
                str(load_path),
                self.embedding,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            # Fallback to old format if model-specific config doesn't exist
            logger.warning(f"Could not load FAISS vector store with model suffix: {e}")
            try:
                self.vector_store = FAISS.load_local(
                    str(load_path), self.embedding, allow_dangerous_deserialization=True
                )
            except Exception as e2:
                logger.warning(f"Could not load FAISS vector store: {e2}")
                self.vector_store = None

        # Load documents
        docs_path = load_path / f"documents_{model_suffix}.pkl"
        # Fallback to old format if model-specific documents don't exist
        if not docs_path.exists():
            docs_path = load_path / "documents.pkl"

        if docs_path.exists():
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)

        logger.info(
            f"Vector store loaded successfully with {len(self.documents)} documents"
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with store statistics
        """
        stats = {
            "total_documents": len(self.documents),
            "embedding_model": self.embedding_model,
            "embedding_provider": self.embedding_provider,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "index_metric": self.index_metric,
            "index_params": self.index_params,
            "vector_store_size": len(self.documents),
        }

        if self.documents:
            # Analyze chunk types
            chunk_types = {}
            for doc in self.documents:
                chunk_type = doc.metadata.get("chunk_type", "unknown")
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            stats["chunk_types"] = chunk_types

        return stats

    def clear(self) -> None:
        """Clear all data from the vector store."""
        logger.info("Clearing vector store")

        # Clear data
        self.vector_store = None
        self.documents = []
        self._index_requires_training = False

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
