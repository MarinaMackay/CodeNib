#!/usr/bin/env python3
"""
Unified SCIP indexer that supports multiple languages (C++, Rust, TypeScript, Python).

This module provides a single SCIPIndexer class that routes to language-specific
indexer implementations based on the 'language' parameter.

Supported languages:
    - 'cpp', 'c++', 'c': C/C++ projects (uses clangd)
    - 'rust', 'rs': Rust projects (uses rust-analyzer)
    - 'ts', 'typescript', 'js', 'javascript': TypeScript/JavaScript projects (uses scip-typescript)
    - 'python', 'py': Python projects (uses scip-python) [default]

Example:
    # C++ indexing
    indexer = SCIPIndexer(project_root="/path/to/cpp/project", language="cpp")
    graph = indexer.run_pipeline()

    # Rust indexing
    indexer = SCIPIndexer(project_root="/path/to/rust/project", language="rust")
    graph = indexer.run_pipeline(exclude_vendored_libraries=True)

    # TypeScript indexing
    indexer = SCIPIndexer(project_root="/path/to/ts/project", language="ts")
    graph = indexer.run_pipeline(infer_tsconfig=True)

    # Python indexing (default)
    indexer = SCIPIndexer(project_root="/path/to/python/project", language="python")
    # Or omit language parameter for backward compatibility
    indexer = SCIPIndexer(project_root="/path/to/python/project")
"""
from pathlib import Path
from typing import List, Optional, Union

from ..graph.code_graph import CodeGraph
from ..log_utils import get_logger
from ..profiler import Profiler

logger = get_logger("scip_indexer")


class SCIPIndexer:
    """
    Unified SCIP indexer that delegates to language-specific implementations.

    This class acts as a router: it selects the appropriate language-specific
    indexer based on the 'language' parameter and delegates all operations to it.
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
        project_root: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        exclude_patterns: Optional[List] = None,
        profiler: Optional[Profiler] = None,
        language: Optional[str] = None,
        idx_directory: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the SCIP indexer.

        Args:
            project_root: Root directory of the project to index
            output_dir: Directory to store output files
            exclude_patterns: List of patterns to exclude from indexing
            profiler: Profiler instance for performance tracking
            language: Programming language ('cpp', 'rust', 'ts', 'python', or None for Python)
            idx_directory: (C/C++ only) Path to directory containing clangd .idx files.

        Raises:
            ValueError: If the language is not supported
        """
        self.project_root = Path(project_root).absolute()

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
        self._delegate = self._create_language_indexer(
            project_root=project_root,
            output_dir=output_dir,
            exclude_patterns=exclude_patterns,
            profiler=profiler,
            idx_directory=idx_directory,
        )

        # Expose delegate's file paths for backward compatibility
        self.output_dir = self._delegate.output_dir
        self.index_file = self._delegate.index_file
        self.decoded_file = self._delegate.decoded_file
        self.graph_file = self._delegate.graph_file
        self.profiler = self._delegate.profiler

        logger.info(f"Initialized SCIP indexer for {self.language} at {self.project_root}")

    def _create_language_indexer(
        self,
        project_root: Union[str, Path],
        output_dir: Optional[Union[str, Path]],
        exclude_patterns: Optional[List],
        profiler: Optional[Profiler],
        idx_directory: Optional[Union[str, Path]] = None,
    ):
        """
        Create the appropriate language-specific indexer.

        Returns:
            Language-specific indexer instance (SCIPIndexerBase subclass)
        """
        if self.language == 'cpp':
            from .clangd_indexer import ClangdIndexer
            return ClangdIndexer(
                project_root=project_root,
                output_dir=output_dir,
                exclude_patterns=exclude_patterns,
                profiler=profiler,
                idx_directory=idx_directory,
            )

        elif self.language == 'rust':
            from .scip_indexer_rust import SCIPRustIndexer
            return SCIPRustIndexer(
                project_root=project_root,
                output_dir=output_dir,
                exclude_patterns=exclude_patterns,
                profiler=profiler,
            )

        elif self.language == 'ts':
            from .scip_indexer_ts import SCIPTypeScriptIndexer
            return SCIPTypeScriptIndexer(
                project_root=project_root,
                output_dir=output_dir,
                exclude_patterns=exclude_patterns,
                profiler=profiler,
            )

        elif self.language == 'python':
            from .scip_indexer_python import SCIPPythonIndexer
            return SCIPPythonIndexer(
                project_root=project_root,
                output_dir=output_dir,
                exclude_patterns=exclude_patterns,
                profiler=profiler,
            )

        else:
            raise ValueError(f"No indexer implementation for language: {self.language}")

    # ── Delegated methods ──────────────────────────────────────────────

    def generate_index(self, **kwargs) -> bool:
        """Generate SCIP index for the project. See language-specific indexer for args."""
        return self._delegate.generate_index(**kwargs)

    def decode_index(self) -> bool:
        """Decode the SCIP index using protobuf."""
        return self._delegate.decode_index()

    def process_index(
        self, output_file: Optional[str] = None
    ) -> Union[CodeGraph, None]:
        """Process the decoded SCIP index into a CodeGraph."""
        return self._delegate.process_index(output_file=output_file)

    def run_pipeline(
        self,
        output_file: Optional[str] = None,
        skip_level: Optional[str] = None,
        *,
        reset_profiler: bool = True,
        report_profile: bool = True,
        **kwargs,
    ) -> Union[CodeGraph, None]:
        """
        Run the complete SCIP indexing pipeline: generate, decode, and process.

        Args:
            output_file: Path to write the processed data to (if None, uses self.graph_file)
            skip_level: Cache/skip level - 'graph', 'decode', 'raw', or None
            reset_profiler: Clear profiler stats before running the pipeline
            report_profile: Emit profiler summary automatically after the run
            **kwargs: Language-specific options passed to generate_index()

        Returns:
            CodeGraph: Processed graph object
        """
        return self._delegate.run_pipeline(
            output_file=output_file,
            skip_level=skip_level,
            reset_profiler=reset_profiler,
            report_profile=report_profile,
            **kwargs,
        )

    def clear_cache(self, level: str = "all") -> bool:
        """Clear cache files at different levels."""
        return self._delegate.clear_cache(level=level)
