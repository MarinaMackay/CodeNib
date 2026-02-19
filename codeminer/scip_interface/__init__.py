"""
SCIP Interface Module

This module provides unified interfaces for SCIP indexing and decoding across multiple languages.

Main Classes:
    - SCIPIndexer: Unified indexer supporting C++, Rust, TypeScript, and Python
    - SCIPDecoder: Unified decoder supporting C++, Rust, TypeScript, and Python

Language-Specific Classes (for advanced use):
    - SCIPClangIndexer, SCIPCppGraphDecoder: C++ specific
    - SCIPRustIndexer, SCIPRustGraphDecoder: Rust specific
    - SCIPTypeScriptIndexer, SCIPTypeScriptGraphDecoder: TypeScript specific

Usage:
    # Unified interface (recommended)
    indexer = SCIPIndexer(project_root="/path/to/project", language="cpp")
    graph = indexer.run_pipeline()

    decoder = SCIPDecoder(index_file_path="/path/to/index.decoded", language="rust")
    graph = decoder.decode()

    # Language-specific interface (advanced)
    indexer = SCIPClangIndexer(project_root="/path/to/cpp/project")
    graph = indexer.run_pipeline()
"""

# Unified interfaces (recommended)
from .scip_decode import SCIPGraphDecoder
from .scip_indexer import SCIPIndexer

# Language-specific indexers (for advanced use)
from .scip_decode_clang import SCIPCppGraphDecoder
from .scip_decode_rust import SCIPRustGraphDecoder
from .scip_decode_ts import SCIPTypeScriptGraphDecoder
from .scip_indexer_base import SCIPIndexerBase
from .scip_indexer_clang import SCIPClangIndexer
from .scip_indexer_rust import SCIPRustIndexer
from .scip_indexer_ts import SCIPTypeScriptIndexer

__all__ = [
    # Unified interfaces
    "SCIPIndexer",
    "SCIPGraphDecoder",
    # Language-specific classes (for advanced use)
    "SCIPIndexerBase",
    "SCIPClangIndexer",
    "SCIPCppGraphDecoder",
    "SCIPRustIndexer",
    "SCIPRustGraphDecoder",
    "SCIPTypeScriptIndexer",
    "SCIPTypeScriptGraphDecoder",
]
