#!/usr/bin/env python3
"""Pytest fixtures shared across test modules."""

import subprocess
from pathlib import Path

import pytest

HTTPIE_REPO_URL = "https://github.com/httpie/cli.git"
HTTPIE_REPO_PATH = Path("/tmp/httpie-cli")


def ensure_httpie_repo() -> Path:
    """Clone the httpie/cli repository if needed and return its path."""
    if not HTTPIE_REPO_PATH.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", HTTPIE_REPO_URL, str(HTTPIE_REPO_PATH)],
            check=True,
        )
    return HTTPIE_REPO_PATH


@pytest.fixture(scope="session")
def httpie_cli_repo():
    """Provide a cached checkout of the httpie/cli repository."""
    return ensure_httpie_repo()
