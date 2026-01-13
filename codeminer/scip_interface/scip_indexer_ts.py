#!/usr/bin/env python3
"""
SCIP indexer for TypeScript and JavaScript projects using scip-typescript.
"""
import subprocess
from pathlib import Path
from typing import List, Optional, Union

from ..log_utils import get_logger
from ..profiler import Profiler
from .scip_indexer_base import SCIPIndexerBase

logger = get_logger("scip_ts_indexer")


class SCIPTypeScriptIndexer(SCIPIndexerBase):
    """
    SCIP indexer for TypeScript and JavaScript projects.

    Uses the scip-typescript tool to generate SCIP indices for TypeScript/JavaScript codebases.
    """

    def __init__(
        self,
        project_root: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        exclude_patterns: Optional[List] = None,
        profiler: Optional[Profiler] = None,
    ):
        """
        Initialize the TypeScript SCIP indexer.

        Args:
            project_root: Root directory of the TypeScript/JavaScript project
            output_dir: Directory to store output files (defaults to /tmp/project_name)
            exclude_patterns: List of patterns to exclude from indexing
            profiler: Profiler instance for performance tracking
        """
        super().__init__(
            project_root=project_root,
            output_dir=output_dir,
            exclude_patterns=exclude_patterns,
            profiler=profiler,
            language="typescript",
        )

    def _check_indexer_available(self) -> bool:
        """
        Check if scip-typescript is available.

        Returns:
            bool: True if scip-typescript is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["scip-typescript", "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"scip-typescript version: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error(
                "scip-typescript not found. Please install it with: npm install -g @sourcegraph/scip-typescript"
            )
            return False

    def _build_index_command(
        self,
        project_name: Optional[str] = None,
        infer_tsconfig: bool = False,
        yarn_workspaces: bool = False,
        pnpm_workspaces: bool = False,
        npm_workspaces: bool = False,
        **kwargs,
    ) -> List[str]:
        """
        Build the command to generate the SCIP index for TypeScript.

        Args:
            project_name: Project name (not used by scip-typescript, kept for compatibility)
            infer_tsconfig: Infer tsconfig for JavaScript projects without tsconfig.json
            yarn_workspaces: Enable Yarn workspaces support
            pnpm_workspaces: Enable pnpm workspaces support
            npm_workspaces: Enable npm workspaces support
            **kwargs: Additional arguments (ignored)

        Returns:
            List[str]: Command as list of strings
        """
        cmd = ["scip-typescript", "index"]

        # Set working directory
        cmd.extend(["--cwd", str(self.project_root)])

        # Output path
        cmd.extend(["--output", str(self.index_file)])

        # Handle workspace options (mutually exclusive)
        if yarn_workspaces:
            cmd.append("--yarn-workspaces")
        elif pnpm_workspaces:
            cmd.append("--pnpm-workspaces")
        elif npm_workspaces:
            cmd.append("--npm-workspaces")

        # Infer tsconfig for JavaScript projects
        if infer_tsconfig:
            cmd.append("--infer-tsconfig")

        return cmd

    def _get_decoder_class(self):
        """
        Get the decoder class for TypeScript.

        Returns:
            SCIPTypeScriptGraphDecoder class
        """
        from .scip_decode_ts import SCIPTypeScriptGraphDecoder

        return SCIPTypeScriptGraphDecoder

    def generate_index(
        self,
        project_name: Optional[str] = None,
        infer_tsconfig: bool = False,
        yarn_workspaces: bool = False,
        pnpm_workspaces: bool = False,
        npm_workspaces: bool = False,
    ) -> bool:
        """
        Generate SCIP index for the TypeScript/JavaScript project.

        Args:
            project_name: Project name (not used by scip-typescript)
            infer_tsconfig: Infer tsconfig for JavaScript projects without tsconfig.json
            yarn_workspaces: Enable Yarn workspaces support
            pnpm_workspaces: Enable pnpm workspaces support
            npm_workspaces: Enable npm workspaces support

        Returns:
            bool: True if index generation was successful, False otherwise
        """
        return super().generate_index(
            project_name=project_name,
            infer_tsconfig=infer_tsconfig,
            yarn_workspaces=yarn_workspaces,
            pnpm_workspaces=pnpm_workspaces,
            npm_workspaces=npm_workspaces,
        )
