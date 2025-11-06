"""
Embedding module for CodeMiner.
Provides vector storage and semantic search capabilities for code chunks.
"""

from .vector_store import CodeVectorStore, create_code_vector_store

__all__ = ["CodeVectorStore", "create_code_vector_store"]
