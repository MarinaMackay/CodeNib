#!/usr/bin/env python3
"""
SCIP indexer for TypeScript and JavaScript projects using scip-typescript.
"""
import json
import shutil
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
                "scip-typescript not found. Please install it with: "
                "npm install -g @sourcegraph/scip-typescript"
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

    def _install_dependencies(self, timeout_sec: int = 600) -> None:
        """
        Install Node dependencies in project root before indexing.

        scip-typescript requires project dependencies to be available for
        reliable indexing, especially for JS projects inferred from package.json.
        """
        package_json = self.project_root / "package.json"
        if not package_json.exists():
            return

        if (self.project_root / "yarn.lock").exists() and shutil.which("yarn"):
            cmd = ["yarn", "install", "--frozen-lockfile"]
        elif (self.project_root / "pnpm-lock.yaml").exists() and shutil.which("pnpm"):
            cmd = ["pnpm", "install", "--frozen-lockfile"]
        elif (self.project_root / "package-lock.json").exists():
            cmd = ["npm", "ci"]
        else:
            cmd = ["npm", "install"]

        logger.info("Installing TypeScript dependencies: %s", " ".join(cmd))
        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except FileNotFoundError:
            logger.warning(
                "Package manager command not found (%s). Continuing without install.",
                cmd[0],
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "Dependency install timed out after %ss. Continuing anyway.",
                timeout_sec,
            )
        except subprocess.CalledProcessError as e:
            logger.warning("Dependency install failed: %s", e)
            if e.stderr:
                logger.warning("install stderr: %s", e.stderr[:500].strip())

    def _normalize_workspace_kwargs(self, kwargs: dict) -> dict:
        """
        Ensure workspace flags are consistent with available package managers.
        """
        yarn_available = bool(shutil.which("yarn"))
        pnpm_available = bool(shutil.which("pnpm"))
        npm_available = bool(shutil.which("npm"))

        if kwargs.get("yarn_workspaces") and not yarn_available:
            logger.warning(
                "yarn_workspaces requested but yarn is not installed; disabling."
            )
            kwargs["yarn_workspaces"] = False
            if npm_available:
                kwargs["npm_workspaces"] = True

        if kwargs.get("pnpm_workspaces") and not pnpm_available:
            logger.warning(
                "pnpm_workspaces requested but pnpm is not installed; disabling."
            )
            kwargs["pnpm_workspaces"] = False
            if npm_available:
                kwargs["npm_workspaces"] = True

        if kwargs.get("npm_workspaces") and not npm_available:
            logger.warning(
                "npm_workspaces requested but npm is not installed; disabling."
            )
            kwargs["npm_workspaces"] = False

        # Keep flags mutually exclusive.
        if kwargs.get("yarn_workspaces"):
            kwargs["pnpm_workspaces"] = False
            kwargs["npm_workspaces"] = False
        elif kwargs.get("pnpm_workspaces"):
            kwargs["npm_workspaces"] = False

        return kwargs

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
        # Guardrail: if no explicit ts/js config exists, force infer mode.
        has_tsconfig = (self.project_root / "tsconfig.json").exists()
        has_jsconfig = (self.project_root / "jsconfig.json").exists()
        if not infer_tsconfig and not has_tsconfig and not has_jsconfig:
            infer_tsconfig = True
            logger.info(
                "No tsconfig.json/jsconfig.json in %s; forcing infer_tsconfig=True",
                self.project_root,
            )

        return super().generate_index(
            project_name=project_name,
            infer_tsconfig=infer_tsconfig,
            yarn_workspaces=yarn_workspaces,
            pnpm_workspaces=pnpm_workspaces,
            npm_workspaces=npm_workspaces,
        )

    def run_pipeline(
        self,
        output_file: Optional[str] = None,
        skip_level: Optional[str] = None,
        *,
        reset_profiler: bool = True,
        report_profile: bool = True,
        **kwargs,
    ):
        """
        Run TypeScript pipeline with language-specific defaults.

        If neither tsconfig.json nor jsconfig.json exists, enable
        --infer-tsconfig automatically unless caller explicitly sets it.
        """
        # Drop Python-specific kwargs that may be forwarded by callers.
        kwargs.pop("target_dir", None)
        kwargs.pop("cwd", None)

        # Auto-select workspace mode when not explicitly provided.
        workspace_flags = ("yarn_workspaces", "pnpm_workspaces", "npm_workspaces")
        has_workspace_mode = any(kwargs.get(flag) for flag in workspace_flags)
        if not has_workspace_mode:
            package_json = self.project_root / "package.json"
            if package_json.exists():
                try:
                    payload = json.loads(package_json.read_text(encoding="utf-8"))
                    has_workspaces = bool(payload.get("workspaces"))
                except Exception:
                    has_workspaces = False
                if has_workspaces:
                    if (
                        self.project_root / "pnpm-workspace.yaml"
                    ).exists() and shutil.which("pnpm"):
                        kwargs["pnpm_workspaces"] = True
                        logger.info("Detected pnpm workspace in %s", self.project_root)
                    elif (self.project_root / "yarn.lock").exists() and shutil.which(
                        "yarn"
                    ):
                        kwargs["yarn_workspaces"] = True
                        logger.info("Detected yarn workspace in %s", self.project_root)
                    else:
                        kwargs["npm_workspaces"] = True
                        logger.info("Detected npm workspace in %s", self.project_root)

        # Forced default for dataset-style repos:
        # scip-typescript frequently fails on root projects without tsconfig.
        if "infer_tsconfig" not in kwargs:
            kwargs["infer_tsconfig"] = True
            logger.info("Enabling infer_tsconfig for %s", self.project_root)

        kwargs = self._normalize_workspace_kwargs(kwargs)
        self._install_dependencies()
        logger.info("TypeScript run_pipeline kwargs: %s", kwargs)
        return super().run_pipeline(
            output_file=output_file,
            skip_level=skip_level,
            reset_profiler=reset_profiler,
            report_profile=report_profile,
            **kwargs,
        )
