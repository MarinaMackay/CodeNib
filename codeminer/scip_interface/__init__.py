"""
SCIP Interface Module

This module provides unified interfaces for SCIP indexing and decoding across multiple languages.

Main Classes:
    - SCIPIndexer: Unified indexer supporting C++, Rust, TypeScript, and Python
    - SCIPGraphDecoder: Unified decoder supporting C++, Rust, TypeScript, and Python

Language-Specific Classes (for advanced use):
    - ClangdIndexer, ClangdGraphDecoder: C++ specific
    - SCIPRustIndexer, SCIPRustGraphDecoder: Rust specific
    - SCIPTypeScriptIndexer, SCIPTypeScriptGraphDecoder: TypeScript specific
    - SCIPPythonIndexer, SCIPPythonGraphDecoder: Python specific

Usage:
    # Unified interface (recommended)
    indexer = SCIPIndexer(project_root="/path/to/project", language="cpp")
    graph = indexer.run_pipeline()

    decoder = SCIPGraphDecoder(index_file_path="/path/to/index.decoded", language="rust")
    graph = decoder.decode()

    # Language-specific interface (advanced)
    indexer = ClangdIndexer(project_root="/path/to/cpp/project")
    graph = indexer.run_pipeline()
"""

# Unified interfaces (recommended)
from .scip_decode import SCIPGraphDecoder
from .scip_indexer import SCIPIndexer

# Language-specific indexers
from .scip_indexer_base import SCIPIndexerBase
from .scip_indexer_python import SCIPPythonIndexer
from .scip_indexer_rust import SCIPRustIndexer
from .scip_indexer_ts import SCIPTypeScriptIndexer
from .clangd_indexer import ClangdIndexer

# Language-specific decoders
from .scip_decode_python import SCIPPythonGraphDecoder
from .scip_decode_rust import SCIPRustGraphDecoder
from .scip_decode_ts import SCIPTypeScriptGraphDecoder
from .clangd_decode import ClangdGraphDecoder


__all__ = [
    # Unified interfaces
    "SCIPIndexer",
    "SCIPGraphDecoder",
    # Base class
    "SCIPIndexerBase",
    # Language-specific indexers
    "SCIPPythonIndexer",
    "ClangdIndexer",
    "SCIPRustIndexer",
    "SCIPTypeScriptIndexer",
    # Language-specific decoders
    "SCIPPythonGraphDecoder",
    "ClangdGraphDecoder",
    "SCIPRustGraphDecoder",
    "SCIPTypeScriptGraphDecoder",
]
