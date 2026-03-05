#!/usr/bin/env python3
"""Pytest fixtures for dataset tests using real SWE-bench Multilingual data."""

from pathlib import Path

import pytest

from codeminer.dataset.gt_locate import GTLocator
from codeminer.dataset.swebench_multilingual import SwebenchMultilingualDataset

# Persistent cache so repos survive across test runs.
GT_TEST_WORK_DIR = Path("/tmp/codeminer-gt-test")

# One representative repo per language we support in gt_locate.
# Chosen for moderate size and sufficient instance count.
REPO_FIXTURES = {
    "go": "caddyserver/caddy",
    "cpp": "redis/redis",
    "rust": "tokio-rs/axum",
    "typescript": "preactjs/preact",
}


@pytest.fixture(scope="session")
def swebench_multilingual_dataset():
    """Load the full SWE-bench Multilingual test split (session-cached)."""
    ds = SwebenchMultilingualDataset(
        dataset="SWE-bench/SWE-bench_Multilingual",
        split="test",
    )
    return ds.load()


def _first_instance_for_repo(dataset, repo: str):
    """Return the first instance whose ``repo`` field matches *repo*."""
    for row in dataset:
        if row["repo"] == repo:
            return dict(row)
    return None


@pytest.fixture(scope="session")
def gt_locator():
    """Shared GTLocator with a persistent work directory."""
    GT_TEST_WORK_DIR.mkdir(parents=True, exist_ok=True)
    return GTLocator(work_dir=str(GT_TEST_WORK_DIR))


@pytest.fixture(scope="session")
def go_instance(swebench_multilingual_dataset):
    inst = _first_instance_for_repo(swebench_multilingual_dataset, REPO_FIXTURES["go"])
    if inst is None:
        pytest.skip("No Go instance found in SWE-bench Multilingual")
    return inst


@pytest.fixture(scope="session")
def cpp_instance(swebench_multilingual_dataset):
    inst = _first_instance_for_repo(swebench_multilingual_dataset, REPO_FIXTURES["cpp"])
    if inst is None:
        pytest.skip("No C++ instance found in SWE-bench Multilingual")
    return inst


@pytest.fixture(scope="session")
def rust_instance(swebench_multilingual_dataset):
    inst = _first_instance_for_repo(
        swebench_multilingual_dataset, REPO_FIXTURES["rust"]
    )
    if inst is None:
        pytest.skip("No Rust instance found in SWE-bench Multilingual")
    return inst


@pytest.fixture(scope="session")
def typescript_instance(swebench_multilingual_dataset):
    inst = _first_instance_for_repo(
        swebench_multilingual_dataset, REPO_FIXTURES["typescript"]
    )
    if inst is None:
        pytest.skip("No TypeScript instance found in SWE-bench Multilingual")
    return inst
