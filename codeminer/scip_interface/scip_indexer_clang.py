#!/usr/bin/env python3
"""
SCIP indexer for C/C++ projects using scip-clang.
"""
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Union

from ..log_utils import get_logger
from ..profiler import Profiler
from .scip_indexer_base import SCIPIndexerBase

logger = get_logger("scip_clang_indexer")


class SCIPClangIndexer(SCIPIndexerBase):
    """
    SCIP indexer for C/C++ projects.

    Uses the scip-clang tool to generate SCIP indices for C/C++ codebases.
    scip-clang requires a JSON compilation database (compile_commands.json).
    """

    def __init__(
        self,
        project_root: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        exclude_patterns: Optional[List] = None,
        profiler: Optional[Profiler] = None,
    ):
        """
        Initialize the Clang SCIP indexer.

        Args:
            project_root: Root directory of the C/C++ project (must contain compile_commands.json)
            output_dir: Directory to store output files (defaults to /tmp/project_name)
            exclude_patterns: List of patterns to exclude from indexing
            profiler: Profiler instance for performance tracking
        """
        super().__init__(
            project_root=project_root,
            output_dir=output_dir,
            exclude_patterns=exclude_patterns,
            profiler=profiler,
            language="clang",
        )

    def _find_scip_clang(self) -> Optional[str]:
        """
        Find the scip-clang executable.

        Searches in the following order:
        1. System PATH (using shutil.which)
        2. Current working directory
        3. Project root directory

        Returns:
            Optional[str]: Path to scip-clang executable, or None if not found
        """
        # Try to find in PATH
        scip_clang = shutil.which("scip-clang")
        if scip_clang:
            return scip_clang

        # Try current working directory
        cwd_scip_clang = Path.cwd() / "scip-clang"
        if cwd_scip_clang.exists() and cwd_scip_clang.is_file():
            return str(cwd_scip_clang.absolute())

        # Try project root
        project_scip_clang = self.project_root / "scip-clang"
        if project_scip_clang.exists() and project_scip_clang.is_file():
            return str(project_scip_clang.absolute())

        return None

    def _check_indexer_available(self) -> bool:
        """
        Check if scip-clang is available.

        Returns:
            bool: True if scip-clang is available, False otherwise
        """
        scip_clang_path = self._find_scip_clang()
        if not scip_clang_path:
            logger.error(
                "scip-clang not found. Please install it from: "
                "https://github.com/sourcegraph/scip-clang\n"
                "Or place the scip-clang binary in the current directory or project root."
            )
            return False

        try:
            result = subprocess.run(
                [scip_clang_path, "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"scip-clang version: {result.stdout.strip()}")
            # Store the path for later use
            self._scip_clang_path = scip_clang_path
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(
                f"Error running scip-clang at {scip_clang_path}: {e}\n"
                "Please install it from: https://github.com/sourcegraph/scip-clang"
            )
            return False

    def _build_index_command(
        self,
        compdb_path: Optional[str] = None,
        show_compiler_diagnostics: bool = False,
        **kwargs,
    ) -> List[str]:
        """
        Build the command to generate the SCIP index for C/C++.

        Args:
            compdb_path: Path to the compilation database (compile_commands.json)
                        Defaults to <project_root>/compile_commands.json or
                        <project_root>/build/compile_commands.json
            show_compiler_diagnostics: Show compiler diagnostics during indexing
            **kwargs: Additional arguments (ignored)

        Returns:
            List[str]: Command as list of strings
        """
        # Use the scip-clang path found during availability check
        scip_clang_cmd = getattr(self, "_scip_clang_path", "scip-clang")
        cmd = [scip_clang_cmd]

        # Determine compilation database path
        if compdb_path:
            comp_db = Path(compdb_path)
        else:
            # Try common locations
            comp_db = self.project_root / "compile_commands.json"
            if not comp_db.exists():
                comp_db = self.project_root / "build" / "compile_commands.json"

        # Compilation database is required
        cmd.extend(["--compdb-path", str(comp_db)])

        # Show compiler diagnostics if requested
        if show_compiler_diagnostics:
            cmd.append("--show-compiler-diagnostics")

        return cmd

    def _get_decoder_class(self):
        """
        Get the decoder class for C/C++.

        Returns:
            SCIPCppGraphDecoder class for C++-specific symbol handling
        """
        from .scip_decode_clang import SCIPCppGraphDecoder

        return SCIPCppGraphDecoder

    def generate_index(
        self,
        compdb_path: Optional[str] = None,
        show_compiler_diagnostics: bool = False,
    ) -> bool:
        """
        Generate SCIP index for the C/C++ project.

        WARNING: You must invoke scip-clang from the project root, not from a subdirectory.

        Args:
            compdb_path: Path to the compilation database (compile_commands.json)
                        Defaults to <project_root>/compile_commands.json or
                        <project_root>/build/compile_commands.json
            show_compiler_diagnostics: Show compiler diagnostics during indexing

        Returns:
            bool: True if index generation was successful, False otherwise
        """
        # Determine compilation database path
        if compdb_path:
            comp_db = Path(compdb_path)
        else:
            # Try common locations
            comp_db = self.project_root / "compile_commands.json"
            if not comp_db.exists():
                comp_db = self.project_root / "build" / "compile_commands.json"

        # Check if compilation database exists
        if not comp_db.exists():
            logger.error(
                f"Compilation database not found. Tried:\n"
                f"  - {self.project_root / 'compile_commands.json'}\n"
                f"  - {self.project_root / 'build' / 'compile_commands.json'}\n\n"
                f"Please generate a compilation database first:\n"
                f"  - CMake: cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON\n"
                f"  - Bazel: Use bazel-compile-commands-extractor\n"
                f"  - Make: Use Bear: bear -- make all\n"
                f"  - See https://github.com/sourcegraph/scip-clang for more options"
            )
            return False

        logger.info(f"Using compilation database: {comp_db}")

        # Run the base class generate_index which handles the actual command execution
        success = super().generate_index(
            compdb_path=str(comp_db),
            show_compiler_diagnostics=show_compiler_diagnostics,
        )

        # scip-clang generates index.scip in the current directory by default
        # We need to move it to the output directory if it's not already there
        if success:
            default_index = self.project_root / "index.scip"
            if default_index.exists() and default_index != self.index_file:
                logger.info(
                    f"Moving index from {default_index} to {self.index_file}"
                )
                default_index.rename(self.index_file)

        return success
