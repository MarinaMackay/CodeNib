"""
Code chunking module for splitting source code files into semantic chunks.
"""

from .base import BaseCodeChunker, CodeChunk
from .cpp_chunker import CppCodeChunker
from .python_chunker import PythonCodeChunker


# Factory function to create appropriate chunker
def create_chunker(language: str, max_lines_per_chunk: int | None = None) -> BaseCodeChunker:
    """
    Create a code chunker for the specified language.

    Args:
        language: Programming language ('python', 'cpp', 'java', etc.)
        max_lines_per_chunk: Optional maximum number of lines per emitted chunk

    Returns:
        Language-specific code chunker instance

    Raises:
        ValueError: If the language is not supported
    """
    language = language.lower()

    if language == "python":
        return PythonCodeChunker(max_lines_per_chunk=max_lines_per_chunk)
    elif language in ("cpp", "c++", "cxx"):
        return CppCodeChunker(max_lines_per_chunk=max_lines_per_chunk)
    else:
        raise ValueError(f"Unsupported language: {language}")


__all__ = [
    "CodeChunk",
    "BaseCodeChunker",
    "PythonCodeChunker",
    "CppCodeChunker",
    "create_chunker",
]
