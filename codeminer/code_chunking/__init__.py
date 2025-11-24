"""
Code chunking module for splitting source code files into semantic chunks.
"""

from .base import BaseCodeChunker, CodeChunk
from .cpp_chunker import CppCodeChunker
from .python_chunker import PythonCodeChunker


# Factory function to create appropriate chunker
def create_chunker(
    language: str,
    max_lines_per_chunk: int | None = None,
    chunk_depth: int = 2,
    include_header_epilogue: bool = False,
    include_class_level: bool = False,
) -> BaseCodeChunker:
    """
    Create a code chunker for the specified language.

    Args:
        language: Programming language ('python', 'cpp', 'java', etc.)
        max_lines_per_chunk: Maximum number of lines per emitted chunk. Default: None (no splitting)
        chunk_depth: Depth of AST traversal (1=top-level only, 2=include methods)
        include_header_epilogue: Whether to include file headers and epilogues. Default: False
        include_class_level: Whether to include class definitions as chunks. Default: False (align with SweRank)

    Returns:
        Language-specific code chunker instance

    Raises:
        ValueError: If the language is not supported
    """
    language = language.lower()

    if language == "python":
        return PythonCodeChunker(
            max_lines_per_chunk=max_lines_per_chunk,
            chunk_depth=chunk_depth,
            include_header_epilogue=include_header_epilogue,
            include_class_level=include_class_level,
        )
    elif language in ("cpp", "c++", "cxx"):
        return CppCodeChunker(
            max_lines_per_chunk=max_lines_per_chunk,
            chunk_depth=chunk_depth,
            include_header_epilogue=include_header_epilogue,
            include_class_level=include_class_level,
        )
    else:
        raise ValueError(f"Unsupported language: {language}")


__all__ = [
    "CodeChunk",
    "BaseCodeChunker",
    "PythonCodeChunker",
    "CppCodeChunker",
    "create_chunker",
]
