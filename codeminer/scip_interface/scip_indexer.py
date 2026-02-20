#!/usr/bin/env python3
"""
Unified SCIP indexer that supports multiple languages (C++, Rust, TypeScript, Python).

This module provides a single SCIPIndexer class that can index projects in multiple
languages by selecting the appropriate language-specific indexer based on the 'language' parameter.

Supported languages:
    - 'cpp', 'c++', 'c': C/C++ projects (uses scip-clang)
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

    # Python indexing (original behavior)
    indexer = SCIPIndexer(project_root="/path/to/python/project", language="python")
    # Or omit language parameter for backward compatibility
    indexer = SCIPIndexer(project_root="/path/to/python/project")
"""
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Union

from ..graph.code_graph import CodeGraph
from ..log_utils import get_logger
from ..profiler import Profiler

logger = get_logger("scip_indexer")


class SCIPIndexer:
    """
    Unified SCIP indexer that supports multiple languages.

    This class can work in two modes:
    1. Language-agnostic mode (default, backward compatible): Acts as Python SCIP indexer
    2. Multi-language mode: Delegates to language-specific indexers based on 'language' parameter
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
            output_dir: Directory to store output files (defaults to /tmp/project_name)
            exclude_patterns: List of patterns to exclude from indexing
            profiler: Profiler instance for performance tracking
            language: Programming language ('cpp', 'rust', 'ts', 'python', or None for Python)
            idx_directory: (C/C++ only) Path to directory containing clangd .idx files.
                          Defaults to <project_root>/.cache/clangd/index/

        Raises:
            ValueError: If the language is not supported
        """
        self.project_root = Path(project_root).absolute()
        self._idx_directory = idx_directory

        # Normalize language name (None means Python for backward compatibility)
        if language is None:
            self.language = 'python'
            self._delegate = None  # Use built-in Python indexer
        else:
            language_lower = language.lower()
            if language_lower not in self.LANGUAGE_ALIASES:
                supported = ', '.join(sorted(set(self.LANGUAGE_ALIASES.keys())))
                raise ValueError(
                    f"Unsupported language: '{language}'. "
                    f"Supported languages: {supported}"
                )
            self.language = self.LANGUAGE_ALIASES[language_lower]

            # Create delegate for non-Python languages
            if self.language != 'python':
                self._delegate = self._create_language_indexer(
                    project_root=project_root,
                    output_dir=output_dir,
                    exclude_patterns=exclude_patterns,
                    profiler=profiler,
                )
                # Copy file paths from delegate
                self.output_dir = self._delegate.output_dir
                self.index_file = self._delegate.index_file
                self.decoded_file = self._delegate.decoded_file
                self.graph_file = self._delegate.graph_file
                self.profiler = self._delegate.profiler
                logger.info(f"Initialized SCIP indexer for {self.language} at {self.project_root}")
                return
            else:
                self._delegate = None

        # Python-specific initialization (original behavior)
        # Set output directory to /tmp/project_name by default
        if output_dir:
            self.output_dir = Path(output_dir).absolute()
        else:
            self.output_dir = Path("/tmp") / self.project_root.name

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Set paths for index files in the output directory
        self.index_file = self.output_dir / "index.scip"
        self.decoded_file = self.output_dir / "index.decoded"
        self.graph_file = self.output_dir / "graph.pkl"
        self.exclude_patterns = exclude_patterns if exclude_patterns else []
        self.profiler = profiler or Profiler("scip_indexer")

        # Path to the conda environment file
        self.module_dir = Path(__file__).parent
        self.env_file = self.module_dir / "scip-environment.yml"
        self.conda_env_name = "scip-env"

    def _create_language_indexer(
        self,
        project_root: Union[str, Path],
        output_dir: Optional[Union[str, Path]],
        exclude_patterns: Optional[List],
        profiler: Optional[Profiler],
    ):
        """
        Create the appropriate language-specific indexer.

        Args:
            project_root: Root directory of the project
            output_dir: Output directory for index files
            exclude_patterns: Patterns to exclude from indexing
            profiler: Profiler instance

        Returns:
            Language-specific indexer instance
        """
        if self.language == 'cpp':
            from .clangd_indexer import ClangdIndexer
            return ClangdIndexer(
                project_root=project_root,
                output_dir=output_dir,
                exclude_patterns=exclude_patterns,
                profiler=profiler,
                idx_directory=self._idx_directory,
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

        else:
            raise ValueError(f"No indexer implementation for language: {self.language}")

    def _ensure_conda_env(self) -> bool:
        """
        Ensure that the conda environment for SCIP is available

        Returns:
            bool: True if environment is available, False otherwise
        """
        try:
            # Check if conda is available
            subprocess.run(["conda", "--version"], check=True, capture_output=True)

            # Check if environment exists
            result = subprocess.run(
                ["conda", "env", "list"], check=True, capture_output=True, text=True
            )

            if self.conda_env_name in result.stdout:
                logger.info(f"Conda environment '{self.conda_env_name}' already exists")
                return True

            # Create the environment if it doesn't exist
            if self.env_file.exists():
                logger.info(f"Creating conda environment '{self.conda_env_name}'...")

                # Use optimized flags to speed up environment creation
                create_cmd = [
                    "conda",
                    "env",
                    "create",
                    "--quiet",
                    "--file",
                    str(self.env_file),
                    "--solver=libmamba",  # Much faster solver
                ]

                try:
                    # First try with the optimized solver
                    subprocess.run(create_cmd, check=True, timeout=300)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    logger.warning(
                        f"Fast environment creation failed: {e}. Falling back to standard method..."
                    )

                    # Fall back to standard conda if the optimized approach fails
                    subprocess.run(
                        ["conda", "env", "create", "--file", str(self.env_file)],
                        check=True,
                    )
                    logger.info(
                        f"Conda environment '{self.conda_env_name}' created successfully"
                    )
                    return True
            else:
                logger.error(f"Environment file not found at {self.env_file}")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Error setting up conda environment: {e}")
            if hasattr(e, "output") and e.output:
                logger.error(f"Command output: {e.output}")
            if hasattr(e, "stderr") and e.stderr:
                logger.error(f"Error details: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error(
                "Conda not found in PATH. Please install conda or add it to PATH."
            )
            return False

    def _run_in_conda_env(
        self, cmd: list, cwd: Optional[Union[str, Path]] = None
    ) -> bool:
        """
        Run a command in the SCIP conda environment

        Args:
            cmd: Command to run
            cwd: Working directory

        Returns:
            bool: True if command succeeded, False otherwise
        """
        if not self._ensure_conda_env():
            return False

        try:
            # Construct the command to run in the conda environment
            conda_cmd = ["conda", "run", "-n", self.conda_env_name] + cmd
            shell = False

            logger.info(f"Running in conda environment: {cmd}")

            # Run the command
            subprocess.run(
                conda_cmd,
                check=True,
                cwd=cwd if cwd else self.project_root,
                shell=shell,
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running command in conda environment: {e}")
            return False

    def generate_index(self, **kwargs) -> bool:
        """
        Generate SCIP index for the project.

        For Python projects (default):
            Args:
                cwd: Working directory to run the index command
                project_name: Project name to use in the index
                target_dir: Optional subdirectory to target for indexing

        For C++ projects (language='cpp'):
            Args:
                compdb_path: Path to compile_commands.json
                show_compiler_diagnostics: Show compiler diagnostics

        For Rust projects (language='rust'):
            Args:
                config_path: Path to JSON configuration file
                exclude_vendored_libraries: Exclude vendored libraries

        For TypeScript projects (language='ts'):
            Args:
                infer_tsconfig: Infer tsconfig for JavaScript projects
                yarn_workspaces: Enable Yarn workspaces support
                pnpm_workspaces: Enable pnpm workspaces support
                npm_workspaces: Enable npm workspaces support

        Returns:
            bool: True if index generation was successful, False otherwise
        """
        # Delegate to language-specific indexer if available
        if self._delegate is not None:
            return self._delegate.generate_index(**kwargs)

        # Python-specific implementation (original behavior)
        cwd = kwargs.get('cwd', ".")
        project_name = kwargs.get('project_name', None)
        target_dir = kwargs.get('target_dir', None)

        cmd = ["scip-python", "index"]

        if cwd:
            cmd.append("--cwd")
            cmd.append(str(Path(cwd).absolute()))

        if project_name:
            cmd.extend(["--project-name", project_name])
        else:
            # Use directory name as project name if not provided
            cmd.extend(["--project-name", self.project_root.name])

        # default output path in /tmp/project_name/index.scip
        cmd.extend(["--output", str(self.index_file)])

        if target_dir:
            cmd.extend(["--target-only", target_dir])

        # Add exclude patterns if any
        for pattern in self.exclude_patterns:
            cmd.extend(["--exclude", pattern])

        logger.debug(f"Running command: {' '.join(cmd)}")

        # Run in conda environment
        with self.profiler.section("generate_index") as section:
            success = self._run_in_conda_env(cmd, self.project_root)
        duration = section.duration

        if success:
            logger.info(f"Successfully generated SCIP index at {self.index_file}")
            logger.info(f"⏱️  Index generation took: {duration:.2f} seconds")
            return True
        else:
            logger.error(f"❌ Index generation failed after {duration:.2f} seconds")
            return False

    def decode_index(self) -> bool:
        """
        Decode the SCIP index using protobuf to create a readable version

        Returns:
            bool: True if decoding was successful, False otherwise
        """
        # Delegate to language-specific indexer if available
        if self._delegate is not None:
            return self._delegate.decode_index()

        # Python-specific implementation (original behavior)
        if not self.index_file.exists():
            logger.error(f"Index file not found at {self.index_file}")
            return False

        try:
            # Using protoc to decode the binary SCIP file
            cmd = [
                "protoc",
                "--decode=scip.Index",
                f"--proto_path={self.module_dir}",
                "scip.proto",
                f"< {self.index_file}",
                f"> {self.decoded_file}",
            ]

            # We need to use shell=True for the redirect operators
            cmd_str = " ".join(cmd)
            logger.info(f"Running command: {cmd_str}")

            with self.profiler.section("decode_index") as section:
                subprocess.run(cmd_str, shell=True, check=True, cwd=self.module_dir)
            duration = section.duration

            logger.info(f"Successfully decoded SCIP index to {self.decoded_file}")
            logger.info(f"⏱️  Index decoding took: {duration:.2f} seconds")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error decoding SCIP index: {e}")
            return False

    def process_index(
        self, output_file: Optional[str] = None
    ) -> Union[CodeGraph, None]:
        """
        Process the decoded SCIP index into a CodeGraph.

        Args:
            output_file: Path to write the processed data to

        Returns:
            CodeGraph: Processed graph object
        """
        # Delegate to language-specific indexer if available
        if self._delegate is not None:
            return self._delegate.process_index(output_file=output_file)

        # Python-specific implementation (original behavior)
        if not self.decoded_file.exists():
            logger.error(f"Decoded index file not found at {self.decoded_file}")
            return None

        try:
            # Import here to avoid circular imports
            from .scip_decode import SCIPGraphDecoder

            # Pass the project root to the decoder to enable directory indexing
            logger.info("Starting SCIP index processing...")
            with self.profiler.section("process_index.decode") as section:
                decoder = SCIPGraphDecoder(
                    str(self.decoded_file), project_root=self.project_root
                )
                graph: CodeGraph = decoder.decode()
            duration = section.duration

            if output_file:
                with self.profiler.section("process_index.save_graph") as save_section:
                    output_path = Path(output_file)
                    decoder.save_graph(str(output_path))
                save_duration = save_section.duration
                logger.info(f"Saved processed SCIP index to {output_path}")
                logger.info(f"⏱️  Graph saving took: {save_duration:.2f} seconds")

            logger.info(f"⏱️  Index processing took: {duration:.2f} seconds")
            return graph

        except Exception as e:
            logger.error(f"Error processing SCIP index: {e}")
            return None

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
                - 'graph': Check if graph.pkl exists, load and return it if found
                - 'decode': Check if index.decoded exists, skip to processing if found
                - 'raw': Check if index.scip exists, skip to decoding if found
                - None: Run full pipeline from scratch (default)
            reset_profiler: Clear profiler stats before running the pipeline
            report_profile: Emit profiler summary automatically after the run
            **kwargs: Language-specific options passed to generate_index()
                For Python: project_name, target_dir
                For C++: compdb_path, show_compiler_diagnostics
                For Rust: config_path, exclude_vendored_libraries
                For TypeScript: infer_tsconfig, yarn_workspaces, pnpm_workspaces, npm_workspaces

        Returns:
            CodeGraph: Processed graph object
        """
        # Delegate to language-specific indexer if available
        if self._delegate is not None:
            return self._delegate.run_pipeline(
                output_file=output_file,
                skip_level=skip_level,
                reset_profiler=reset_profiler,
                report_profile=report_profile,
                **kwargs,
            )

        # Python-specific implementation (original behavior)
        project_name = kwargs.get('project_name', None)
        target_dir = kwargs.get('target_dir', None)
        # Use default graph file if output_file not specified
        if output_file is None:
            output_file = str(self.graph_file)

        # Check graph cache if skip_level is 'graph'
        if skip_level == "graph" and self.graph_file.exists():
            logger.info(f"Loading cached graph from {self.graph_file}")
            try:
                graph = CodeGraph.load_graph(str(self.graph_file))
                logger.info(
                    f"✅ Successfully loaded cached graph ({len(graph.graph.vs)} nodes, {len(graph.graph.es)} edges)"
                )
                return graph
            except Exception as e:
                logger.warning(
                    f"Failed to load cached graph: {e}. Proceeding with pipeline..."
                )

        # Determine what needs to be generated based on what exists
        # Priority: if decoded exists, we don't need index; if graph exists (handled above), we don't need anything

        # Check if we can skip to processing (decoded file exists)
        if skip_level in ("graph", "decode") and self.decoded_file.exists():
            logger.info(
                f"Found existing decoded file at {self.decoded_file}, skipping index generation and decode"
            )
            should_generate_index = False
            should_decode_index = False
        # Check if we can skip to decoding (raw index file exists)
        elif skip_level in ("graph", "decode", "raw") and self.index_file.exists():
            logger.info(
                f"Found existing raw index at {self.index_file}, skipping generation"
            )
            should_generate_index = False
            should_decode_index = True
        # Otherwise, run from scratch
        else:
            should_generate_index = True
            should_decode_index = True

        if reset_profiler:
            self.profiler.reset()

        try:
            # Generate the index if needed
            if should_generate_index:
                logger.info("Generating SCIP index")
                if not self.generate_index(
                    cwd=self.project_root,
                    project_name=project_name,
                    target_dir=target_dir,
                ):
                    return None

            # Decode the index if needed
            if should_decode_index:
                if not self.index_file.exists():
                    logger.error(
                        f"Index file not found at {self.index_file}, cannot decode"
                    )
                    return None
                logger.info("Decoding SCIP index")
                if not self.decode_index():
                    return None

            # Process the index and save graph
            graph = self.process_index(output_file)

            if graph:
                logger.info(
                    f"✅ Graph created successfully ({len(graph.graph.vs)} nodes, {len(graph.graph.es)} edges)"
                )

            return graph
        finally:
            if report_profile:
                self.profiler.report(reset=reset_profiler)

    def clear_cache(self, level: str = "all") -> bool:
        """
        Clear cache files at different levels.

        Args:
            level: Cache level to clear
                - 'graph': Keep only graph.pkl, remove index.decoded and index.scip
                - 'decode': Keep only index.decoded, remove graph.pkl and index.scip
                - 'raw': Keep only index.scip, remove graph.pkl and index.decoded
                - 'all': Remove all cache files (default)

        Returns:
            bool: True if cache clearing was successful, False otherwise
        """
        # Delegate to language-specific indexer if available
        if self._delegate is not None:
            return self._delegate.clear_cache(level=level)

        # Python-specific implementation (original behavior)
        try:
            files_to_remove = []

            if level == "graph":
                # Keep graph.pkl, remove everything else
                files_to_remove = [self.index_file, self.decoded_file]
                logger.info("Clearing cache: keeping graph.pkl only")
            elif level == "decode":
                # Keep index.decoded, remove everything else
                files_to_remove = [self.index_file, self.graph_file]
                logger.info("Clearing cache: keeping index.decoded only")
            elif level == "raw":
                # Keep index.scip, remove everything else
                files_to_remove = [self.decoded_file, self.graph_file]
                logger.info("Clearing cache: keeping index.scip only")
            elif level == "all":
                # Remove all cache files
                files_to_remove = [self.index_file, self.decoded_file, self.graph_file]
                logger.info("Clearing all cache files")
            else:
                logger.error(
                    f"Invalid cache level: {level}. Must be 'graph', 'decode', 'raw', or 'all'"
                )
                return False

            # Remove the specified files
            for file_path in files_to_remove:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Removed {file_path}")
                else:
                    logger.debug(f"File does not exist, skipping: {file_path}")

            logger.info("✅ Cache cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
