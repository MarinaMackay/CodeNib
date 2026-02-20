#!/usr/bin/env python3
"""
Base class for SCIP indexers across different languages.
"""
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

from ..graph.code_graph import CodeGraph
from ..log_utils import get_logger
from ..profiler import Profiler

logger = get_logger("scip_indexer_base")


class SCIPIndexerBase(ABC):
    """
    Abstract base class for SCIP indexers.

    This class provides common functionality for all SCIP indexers,
    while allowing language-specific implementations to customize
    the indexing process.
    """

    def __init__(
        self,
        project_root: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        exclude_patterns: Optional[List] = None,
        profiler: Optional[Profiler] = None,
        language: str = "unknown",
    ):
        """
        Initialize the SCIP indexer.

        Args:
            project_root: Root directory of the project to index
            output_dir: Directory to store output files (defaults to /tmp/project_name)
            exclude_patterns: List of patterns to exclude from indexing
            profiler: Profiler instance for performance tracking
            language: Language being indexed (for logging)
        """
        self.project_root = Path(project_root).absolute()
        self.language = language

        # Set output directory to ~/.codeminer/project_name by default
        if output_dir:
            self.output_dir = Path(output_dir).absolute()
        else:
            self.output_dir = Path.home() / ".codeminer" / self.project_root.name

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Set paths for index files in the output directory
        self.index_file = self.output_dir / "index.scip"
        self.decoded_file = self.output_dir / "index.decoded"
        self.graph_file = self.output_dir / "graph.pkl"
        self.exclude_patterns = exclude_patterns if exclude_patterns else []
        self.profiler = profiler or Profiler(f"scip_{language}_indexer")

        # Path to the scip.proto file (shared across all indexers)
        self.module_dir = Path(__file__).parent
        self.proto_file = self.module_dir / "scip.proto"

    @abstractmethod
    def _check_indexer_available(self) -> bool:
        """
        Check if the language-specific indexer tool is available.

        Returns:
            bool: True if indexer is available, False otherwise
        """
        pass

    @abstractmethod
    def _build_index_command(self, **kwargs) -> List[str]:
        """
        Build the command to generate the SCIP index.

        Args:
            **kwargs: Language-specific options

        Returns:
            List[str]: Command as list of strings
        """
        pass

    @abstractmethod
    def _get_decoder_class(self):
        """
        Get the decoder class for this language.

        Returns:
            The decoder class to use for processing the SCIP index
        """
        pass

    def generate_index(self, **kwargs) -> bool:
        """
        Generate SCIP index for the project.

        Args:
            **kwargs: Language-specific options

        Returns:
            bool: True if index generation was successful, False otherwise
        """
        if not self._check_indexer_available():
            return False

        # Build the command
        cmd = self._build_index_command(**kwargs)

        logger.debug(f"Running command: {' '.join(cmd)}")

        # Run the command
        with self.profiler.section("generate_index") as section:
            try:
                # For Rust projects, override toolchain to use stable rust-analyzer
                # This prevents issues with old toolchains that don't include rust-analyzer
                env = None
                if self.language == "rust":
                    import os
                    env = os.environ.copy()
                    env["RUSTUP_TOOLCHAIN"] = "stable"

                subprocess.run(
                    cmd,
                    check=True,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                success = True
            except subprocess.CalledProcessError as e:
                logger.error(f"Error generating SCIP index: {e}")
                if e.stdout:
                    logger.error(f"stdout: {e.stdout}")
                if e.stderr:
                    logger.error(f"stderr: {e.stderr}")
                success = False

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
        Decode the SCIP index using protobuf to create a readable version.

        Returns:
            bool: True if decoding was successful, False otherwise
        """
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
        Process the decoded SCIP index into a more usable format.

        Args:
            output_file: Path to write the processed data to

        Returns:
            CodeGraph: Processed graph object
        """
        if not self.decoded_file.exists():
            logger.error(f"Decoded index file not found at {self.decoded_file}")
            return None

        try:
            # Get the decoder class for this language
            decoder_class = self._get_decoder_class()

            # Pass the project root to the decoder to enable directory indexing
            logger.info("Starting SCIP index processing...")
            with self.profiler.section("process_index.decode") as section:
                decoder = decoder_class(
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
            **kwargs: Language-specific options passed to generate_index

        Returns:
            CodeGraph: Processed graph object
        """
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
        if skip_level in ("graph", "decode") and self.decoded_file.exists():
            logger.info(
                f"Found existing decoded file at {self.decoded_file}, skipping index generation and decode"
            )
            should_generate_index = False
            should_decode_index = False
        elif skip_level in ("graph", "decode", "raw") and self.index_file.exists():
            logger.info(
                f"Found existing raw index at {self.index_file}, skipping generation"
            )
            should_generate_index = False
            should_decode_index = True
        else:
            should_generate_index = True
            should_decode_index = True

        if reset_profiler:
            self.profiler.reset()

        try:
            # Generate the index if needed
            if should_generate_index:
                logger.info("Generating SCIP index")
                if not self.generate_index(**kwargs):
                    # The indexer reported failure (non-zero exit), but the index file may still have been written 
                    # (e.g. scip-typescript crashes during cleanup after emitting index.scip).
                    # Continue if the file exists.
                    if not self.index_file.exists():
                        return None
                    logger.warning(
                        "Index generation returned failure but %s exists; continuing.",
                        self.index_file,
                    )

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
        try:
            files_to_remove = []

            if level == "graph":
                files_to_remove = [self.index_file, self.decoded_file]
                logger.info("Clearing cache: keeping graph.pkl only")
            elif level == "decode":
                files_to_remove = [self.index_file, self.graph_file]
                logger.info("Clearing cache: keeping index.decoded only")
            elif level == "raw":
                files_to_remove = [self.decoded_file, self.graph_file]
                logger.info("Clearing cache: keeping index.scip only")
            elif level == "all":
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