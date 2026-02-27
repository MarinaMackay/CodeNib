"""
Language-Server Index Module (non-SCIP)

C/C++ indexing and decoding via clangd .idx files.
"""

from .clangd_decode import ClangdGraphDecoder
from .clangd_indexer import ClangdIndexer

__all__ = [
    "ClangdIndexer",
    "ClangdGraphDecoder",
]
