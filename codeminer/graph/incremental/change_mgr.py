"""Change manager: git-based change detection for incremental updates."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

from ...log_utils import get_logger

logger = get_logger(__name__)

_FULL_SHA_RE = re.compile(r"[0-9a-f]{40}")


def _shorten_ref(ref: str) -> str:
    """Shorten a git ref to 12 chars only if it is a full 40-char hex SHA."""
    if _FULL_SHA_RE.fullmatch(ref):
        return ref[:12]
    return ref


def detect_changed_files(
    project_root: str,
    base_commit: str,
    target_commit: str = "HEAD",
    extensions: Optional[set[str]] = None,
) -> dict:
    """Detect changed files between two commits using git diff.

    Args:
        project_root: Path to the git repository root.
        base_commit: Base commit hash or ref.
        target_commit: Target commit hash or ref (default HEAD).
        extensions: Only include files with these extensions.

    Returns:
        Dict with keys: modified, added, deleted, renamed.
        Each value is a list of relative file paths (renamed is list of
        (old_path, new_path) tuples).
    """
    result = {"modified": [], "added": [], "deleted": [], "renamed": []}
    try:
        output = subprocess.check_output(
            [
                "git",
                "diff",
                "--name-status",
                _shorten_ref(base_commit),
                _shorten_ref(target_commit),
            ],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"git diff failed: {e}")
        return result

    for line in output.strip().splitlines():
        parts = line.split("\t")
        if not parts:
            continue

        status = parts[0]
        if status == "M" and len(parts) >= 2:
            path = parts[1]
            if extensions is None or Path(path).suffix in extensions:
                result["modified"].append(path)
        elif status == "A" and len(parts) >= 2:
            path = parts[1]
            if extensions is None or Path(path).suffix in extensions:
                result["added"].append(path)
        elif status == "D" and len(parts) >= 2:
            path = parts[1]
            if extensions is None or Path(path).suffix in extensions:
                result["deleted"].append(path)
        elif status.startswith("R") and len(parts) >= 3:
            old_path, new_path = parts[1], parts[2]
            if extensions is None or Path(new_path).suffix in extensions:
                result["renamed"].append((old_path, new_path))

    return result


def get_changed_line_ranges(
    project_root: str,
    file_path: str,
    base_commit: str,
    target_commit: str = "HEAD",
) -> list[tuple[int, int]]:
    """Get changed line ranges between two commits for a single file.

    Uses ``git diff -U0`` to extract exact changed line ranges in the
    target (new) version of the file.

    Args:
        project_root: Path to the git repository root.
        file_path: Relative path to the file within the repo.
        base_commit: Base commit hash or ref.
        target_commit: Target commit hash or ref (default HEAD).

    Returns:
        List of (start_line, end_line) tuples (0-indexed, inclusive).
    """
    try:
        output = subprocess.check_output(
            [
                "git",
                "diff",
                "-U0",
                _shorten_ref(base_commit),
                _shorten_ref(target_commit),
                "--",
                file_path,
            ],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        logger.debug(f"git diff failed for {file_path}")
        return []

    ranges = []
    for match in re.finditer(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", output):
        start = int(match.group(1)) - 1  # Convert to 0-indexed
        count = int(match.group(2)) if match.group(2) else 1
        if count == 0:
            # Pure deletion: mark the deletion point
            ranges.append((max(0, start - 1), max(0, start - 1)))
            continue
        end = start + count - 1
        ranges.append((start, end))
    return ranges
