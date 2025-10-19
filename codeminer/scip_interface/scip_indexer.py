#!/usr/bin/env python3
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Union

from ..graph.code_graph import CodeGraph
from ..log_utils import get_logger

logger = get_logger("scip_indexer")


class SCIPIndexer:
    """Interface for working with the SCIP index using scip-python"""

    def __init__(
        self,
        project_root: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        exclude_patterns: Optional[List] = None,
    ):
        self.project_root = Path(project_root).absolute()

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

        # Path to the conda environment file
        self.module_dir = Path(__file__).parent
        self.env_file = self.module_dir / "scip-environment.yml"
        self.conda_env_name = "scip-env"

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

    def generate_index(
        self,
        cwd: Union[str, Path] = ".",
        project_name: Optional[str] = None,
        target_dir: Optional[str] = None,
    ) -> bool:
        """
        Generate SCIP index for the project

        Args:
            cwd: Working directory to run the index command
            project_name: Project name to use in the index
            target_dir: Optional subdirectory to target for indexing

        Returns:
            bool: True if index generation was successful, False otherwise
        """
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

        # Time the index generation
        start_time = time.time()

        # Run in conda environment
        success = self._run_in_conda_env(cmd, self.project_root)

        end_time = time.time()
        duration = end_time - start_time

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

            # Time the decoding
            start_time = time.time()

            subprocess.run(cmd_str, shell=True, check=True, cwd=self.module_dir)

            end_time = time.time()
            duration = end_time - start_time

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
        Process the decoded SCIP index into a more usable format

        Args:
            output_file: Path to write the processed data to

        Returns:
            CodeGraph: Processed graph object
        """
        if not self.decoded_file.exists():
            logger.error(f"Decoded index file not found at {self.decoded_file}")
            return None

        try:
            # Import here to avoid circular imports
            from .scip_decode import SCIPGraphDecoder

            # Time the processing
            start_time = time.time()

            # Pass the project root to the decoder to enable directory indexing
            logger.info("Starting SCIP index processing...")
            decoder = SCIPGraphDecoder(
                str(self.decoded_file), project_root=self.project_root
            )
            graph: CodeGraph = decoder.decode()

            end_time = time.time()
            duration = end_time - start_time

            if output_file:
                save_start = time.time()
                output_path = Path(output_file)
                decoder.save_graph(str(output_path))
                save_duration = time.time() - save_start
                logger.info(f"Saved processed SCIP index to {output_path}")
                logger.info(f"⏱️  Graph saving took: {save_duration:.2f} seconds")

            logger.info(f"⏱️  Index processing took: {duration:.2f} seconds")
            return graph

        except Exception as e:
            logger.error(f"Error processing SCIP index: {e}")
            return None

    def run_pipeline(
        self,
        project_name: Optional[str] = None,
        target_dir: Optional[str] = None,
        output_file: Optional[str] = None,
        skip_level: Optional[str] = None,
    ) -> Union[CodeGraph, None]:
        """
        Run the complete SCIP indexing pipeline: generate, decode, and process

        Args:
            project_name: Project name to use in the index
            target_dir: Optional subdirectory to target for indexing
            output_file: Path to write the processed data to (if None, uses self.graph_file)
            skip_level: Cache/skip level - 'graph', 'decode', 'raw', or None
                - 'graph': Check if graph.pkl exists, load and return it if found
                - 'decode': Check if index.decoded exists, skip to processing if found
                - 'raw': Check if index.scip exists, skip to decoding if found
                - None: Run full pipeline from scratch (default)

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

        # Generate the index if needed
        if should_generate_index:
            logger.info(f"Generating SCIP index")
            if not self.generate_index(
                cwd=self.project_root, project_name=project_name, target_dir=target_dir
            ):
                return None

        # Decode the index if needed
        if should_decode_index:
            if not self.index_file.exists():
                logger.error(
                    f"Index file not found at {self.index_file}, cannot decode"
                )
                return None
            logger.info(f"Decoding SCIP index")
            if not self.decode_index():
                return None

        # Process the index and save graph
        graph = self.process_index(output_file)

        if graph:
            logger.info(
                f"✅ Graph created successfully ({len(graph.graph.vs)} nodes, {len(graph.graph.es)} edges)"
            )

        return graph

    def clear_cache(self, level: str = "all") -> bool:
        """
        Clear cache files at different levels

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
