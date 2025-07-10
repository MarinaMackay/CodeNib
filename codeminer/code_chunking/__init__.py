"""
Code chunking module for splitting source code files into semantic chunks.
"""

from .base import BaseCodeChunker, CodeChunk
from .cpp_chunker import CppCodeChunker
from .python_chunker import PythonCodeChunker


# Factory function to create appropriate chunker
def create_chunker(language: str) -> BaseCodeChunker:
    """
    Create a code chunker for the specified language.

    Args:
        language: Programming language ('python', 'cpp', 'java', etc.)

    Returns:
        Language-specific code chunker instance

    Raises:
        ValueError: If the language is not supported
    """
    language = language.lower()

    if language == "python":
        return PythonCodeChunker()
    elif language in ("cpp", "c++", "cxx"):
        return CppCodeChunker()
    else:
        raise ValueError(f"Unsupported language: {language}")


__all__ = [
    "CodeChunk",
    "BaseCodeChunker",
    "PythonCodeChunker",
    "CppCodeChunker",
    "create_chunker",
]
