#!/usr/bin/env python3
"""
SCIP indexer for C/C++ projects using scip-clang.
"""
import json
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
            project_root: Root directory of the C/C++ project
                (must contain compile_commands.json)
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

        # Check if compilation database exists and is valid
        if not self._is_valid_compdb(comp_db):
            logger.error(
                f"Compilation database missing or invalid. Tried:\n"
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
                logger.info(f"Moving index from {default_index} to {self.index_file}")
                default_index.rename(self.index_file)

        return success

    def _auto_generate_compdb(self) -> Optional[Path]:
        """
        Try generating compile_commands.json with common build flows.
        Returns the path if successful, otherwise None.
        """
        # 1) CMake projects
        compdb = self._auto_generate_compdb_cmake()
        if compdb is not None:
            return compdb

        # 2) Make/Autotools projects via Bear
        return self._auto_generate_compdb_bear()

    def _is_valid_compdb(self, compdb: Path) -> bool:
        """Check compile_commands.json is parseable and non-empty."""
        if not compdb.exists() or not compdb.is_file():
            return False
        try:
            if compdb.stat().st_size < 4:
                return False
            with compdb.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            return isinstance(payload, list) and len(payload) > 0
        except Exception:
            return False

    def _auto_generate_compdb_cmake(self) -> Optional[Path]:
        cmake_lists = self.project_root / "CMakeLists.txt"
        if not cmake_lists.exists() or not shutil.which("cmake"):
            return None

        build_dir = self.project_root / "build"
        build_dir.mkdir(exist_ok=True)

        cmd = [
            "cmake",
            "-S",
            str(self.project_root),
            "-B",
            str(build_dir),
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ]
        logger.info("Attempting CMake compilation DB generation: %s", " ".join(cmd))
        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to generate compile_commands.json with CMake: %s", e)
            if e.stderr:
                logger.warning("cmake stderr: %s", e.stderr.strip())
            return None

        compdb = build_dir / "compile_commands.json"
        return compdb if compdb.exists() else None

    def _auto_generate_compdb_bear(self) -> Optional[Path]:
        if not shutil.which("bear"):
            logger.info("bear not found; skipping Make-based compilation DB generation")
            return None
        if not shutil.which("make"):
            logger.info("make not found; cannot run bear -- make")
            return None

        logger.info("Attempting Make compilation DB generation with bear")
        success = self._run_build_command(["bear", "--", "make", "-j"])
        if not success:
            # Fall back to non-parallel in case jobs trigger flaky build scripts.
            success = self._run_build_command(["bear", "--", "make"])
        if not success:
            # Some autotools repos have stale/incomplete Makefiles after checkout.
            # Try bootstrapping and configuring, then retry.
            logger.info("Initial bear+make failed; attempting autotools bootstrap")
            self._bootstrap_autotools()
            # Clear stale artifacts before rebuilding (best effort).
            self._run_build_command(["make", "clean"])
            success = self._run_build_command(["bear", "--", "make", "-j"])
            if not success:
                success = self._run_build_command(["bear", "--", "make"])
        if not success:
            # Last resort: look for Makefile in well-known subdirectories
            # (e.g. micropython has ports/unix/Makefile instead of top-level)
            compdb = self._try_subdirectory_make()
            if compdb:
                return compdb
            return None

        for candidate in (
            self.project_root / "compile_commands.json",
            self.project_root / "build" / "compile_commands.json",
        ):
            if candidate.exists():
                return candidate
        return None

    def _try_subdirectory_make(self) -> Optional[Path]:
        """Try building from a subdirectory that has a Makefile.

        Some projects (e.g. micropython) keep the actual build in a
        subdirectory like ports/unix/.  We search a small set of known
        patterns and attempt ``bear -- make`` there, copying the resulting
        compile_commands.json to the project root so scip-clang can find it.
        """
        candidates = [
            self.project_root / "ports" / "unix",
            self.project_root / "src",
        ]
        for subdir in candidates:
            makefile = subdir / "Makefile"
            if not makefile.exists():
                continue
            logger.info(
                "Found Makefile in subdirectory %s, attempting bear -- make there",
                subdir,
            )
            # Clean stale build artifacts from previous commits sharing this repo dir.
            self._run_build_command(["make", "-C", str(subdir), "clean"])
            # micropython ports/unix needs submodules
            self._run_build_command(["make", "-C", str(subdir), "submodules"])
            if self._run_build_command(
                ["bear", "--", "make", "-C", str(subdir), "-j"]
            ) or self._run_build_command(
                ["bear", "--", "make", "-C", str(subdir)]
            ):
                # bear writes compile_commands.json in cwd (project root)
                compdb = self.project_root / "compile_commands.json"
                if compdb.exists():
                    with open(compdb) as f:
                        entries = json.load(f)
                    if entries:
                        logger.info(
                            "Generated compile_commands.json from %s (%d entries)",
                            subdir,
                            len(entries),
                        )
                        return compdb
        return None

    def _bootstrap_autotools(self) -> None:
        autogen = self.project_root / "autogen.sh"
        configure = self.project_root / "configure"
        configure_ac = self.project_root / "configure.ac"

        # Initialize git submodules if .gitmodules exists (e.g. jq's oniguruma)
        gitmodules = self.project_root / ".gitmodules"
        if gitmodules.exists():
            logger.info("Initializing git submodules")
            self._run_build_command(["git", "submodule", "update", "--init"])

        if autogen.exists():
            logger.info("Running autogen.sh")
            self._run_build_command(["sh", str(autogen)])
        elif configure_ac.exists() and shutil.which("autoreconf"):
            logger.info("Running autoreconf -fi")
            self._run_build_command(["autoreconf", "-fi"])

        if configure.exists():
            configure_cmd = ["sh", str(configure)]
            # jq recommends builtin oniguruma when building from source.
            repo_hint = self.project_root.name.lower()
            if repo_hint in {"jq", "jqlang_jq"}:
                configure_cmd.append("--with-oniguruma=builtin")
            logger.info("Running configure command: %s", " ".join(configure_cmd))
            self._run_build_command(configure_cmd)

    def _run_build_command(self, cmd: List[str], timeout_sec: int = 1200) -> bool:
        logger.info("Running build command: %s", " ".join(cmd))
        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            return True
        except subprocess.TimeoutExpired:
            logger.warning(
                "Command timed out after %ss: %s", timeout_sec, " ".join(cmd)
            )
            return False
        except subprocess.CalledProcessError as e:
            logger.warning("Command failed: %s", " ".join(cmd))
            if e.stderr:
                logger.warning("stderr: %s", e.stderr[:600].strip())
            return False

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
        Run C/C++ pipeline with compile_commands.json auto-discovery/generation.
        """
        # Drop Python-specific kwargs that may be forwarded by callers.
        kwargs.pop("project_name", None)
        kwargs.pop("target_dir", None)
        kwargs.pop("cwd", None)

        if "compdb_path" not in kwargs or not kwargs["compdb_path"]:
            for candidate in (
                self.project_root / "compile_commands.json",
                self.project_root / "build" / "compile_commands.json",
            ):
                if self._is_valid_compdb(candidate):
                    kwargs["compdb_path"] = str(candidate)
                    break
                if candidate.exists():
                    logger.warning(
                        "Ignoring invalid compilation database: %s", candidate
                    )
            else:
                generated = self._auto_generate_compdb()
                if generated is not None and self._is_valid_compdb(generated):
                    kwargs["compdb_path"] = str(generated)
                elif generated is not None:
                    logger.warning(
                        "Auto-generated compilation database is invalid: %s", generated
                    )

        return super().run_pipeline(
            output_file=output_file,
            skip_level=skip_level,
            reset_profiler=reset_profiler,
            report_profile=report_profile,
            **kwargs,
        )
