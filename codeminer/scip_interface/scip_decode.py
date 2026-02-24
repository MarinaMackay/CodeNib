"""
Unified SCIP decoder that supports multiple languages (C++, Rust, TypeScript, Python).

This module provides a single SCIPGraphDecoder class that routes to language-specific
decoder implementations based on the 'language' parameter.

Supported languages:
    - 'cpp', 'c++', 'c': C/C++ projects
    - 'rust', 'rs': Rust projects
    - 'ts', 'typescript', 'js', 'javascript': TypeScript/JavaScript projects
    - 'python', 'py': Python projects [default]

Example:
    # C++ decoder
    decoder = SCIPGraphDecoder(index_file_path="path/to/index.decoded", language="cpp")
    graph = decoder.decode()

    # Rust decoder
    decoder = SCIPGraphDecoder(index_file_path="path/to/index.decoded", language="rust")
    graph = decoder.decode()

    # Python decoder (default)
    decoder = SCIPGraphDecoder(index_file_path="path/to/index.decoded", language="python")
    # Or omit language parameter for backward compatibility
    decoder = SCIPGraphDecoder(index_file_path="path/to/index.decoded")
"""
from typing import Optional

from ..log_utils import get_logger

logger = get_logger("scip_decode")


class SCIPGraphDecoder:
    """
    Unified SCIP decoder that delegates to language-specific implementations.

    This class acts as a router: it selects the appropriate language-specific
    decoder based on the 'language' parameter and delegates all operations to it.
    """

    # Language aliases for flexibility
    LANGUAGE_ALIASES = {
        'cpp': 'cpp',
        'c++': 'cpp',
        'c': 'cpp',
        'rust': 'rust',
        'rs': 'rust',
        'typescript': 'ts',
        'ts': 'ts',
        'javascript': 'ts',
        'js': 'ts',
        'python': 'python',
        'py': 'python',
    }

    def __init__(
        self,
        index_file_path: str,
        project_root: Optional[str] = None,
        language: Optional[str] = None,
    ):
        """
        Initialize the SCIP decoder.

        Args:
            index_file_path: Path to the decoded SCIP index file
            project_root: Root directory of the project (optional)
            language: Programming language ('cpp', 'rust', 'ts', 'python', or None for Python)

        Raises:
            ValueError: If the language is not supported
        """
        self.index_file_path = index_file_path
        self.project_root = project_root

        # Normalize language name (None means Python for backward compatibility)
        if language is None:
            self.language = 'python'
        else:
            language_lower = language.lower()
            if language_lower not in self.LANGUAGE_ALIASES:
                supported = ', '.join(sorted(set(self.LANGUAGE_ALIASES.keys())))
                raise ValueError(
                    f"Unsupported language: '{language}'. "
                    f"Supported languages: {supported}"
                )
            self.language = self.LANGUAGE_ALIASES[language_lower]

        # Create the language-specific delegate
        self._delegate = self._create_language_decoder(
            index_file_path=index_file_path,
            project_root=project_root,
        )

        # Expose delegate's code_graph for backward compatibility
        self.code_graph = self._delegate.code_graph

    def _create_language_decoder(
        self,
        index_file_path: str,
        project_root: Optional[str],
    ):
        """
        Create the appropriate language-specific decoder.

        Returns:
            Language-specific decoder instance
        """
        if self.language == 'cpp':
            from .scip_decode_clang import SCIPCppGraphDecoder
            return SCIPCppGraphDecoder(
                index_file_path=index_file_path,
                project_root=project_root,
            )

        elif self.language == 'rust':
            from .scip_decode_rust import SCIPRustGraphDecoder
            return SCIPRustGraphDecoder(
                index_file_path=index_file_path,
                project_root=project_root,
            )

        elif self.language == 'ts':
            from .scip_decode_ts import SCIPTypeScriptGraphDecoder
            return SCIPTypeScriptGraphDecoder(
                index_file_path=index_file_path,
                project_root=project_root,
            )

        elif self.language == 'python':
            from .scip_decode_python import SCIPPythonGraphDecoder
            return SCIPPythonGraphDecoder(
                index_file_path=index_file_path,
                project_root=project_root,
            )

        else:
            raise ValueError(f"No decoder implementation for language: {self.language}")

    # ── Delegated methods ──────────────────────────────────────────────

    def decode(self):
        """Decode the SCIP index into a CodeGraph."""
        return self._delegate.decode()

    def save_graph(self, output_path: str):
        """Save the decoded graph to a file."""
        return self._delegate.save_graph(output_path)
