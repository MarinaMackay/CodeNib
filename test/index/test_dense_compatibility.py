#!/usr/bin/env python3
"""Test to check if all dense chunk node_ids exist in BM25 graph nodes."""

import warnings
from pathlib import Path

import pytest

from codeminer.code_chunker import CodeChunker
from codeminer.index import BM25CodeIndexer
from codeminer.scip_interface import SCIPIndexer


@pytest.fixture(scope="module")
def repo_paths():
    repo_path = Path(__file__).parent.parent / "simple_repo"
    if not repo_path.exists():
        pytest.skip(f"Test repository not found at {repo_path}")
    output_path = Path.home() / ".codeminer" / "simple_repo_nodes_test"
    return repo_path, output_path


def test_dense_compatibility(repo_paths):
    """Test that all dense chunk node_ids exist in BM25 graph nodes."""
    repo_path, output_path = repo_paths

    try:
        repo_indexer = SCIPIndexer(repo_path, output_dir=output_path)
        graph = repo_indexer.run_pipeline(project_name="simple_repo_compat")
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"BM25 nodes creation failed: {exc}")

    if not graph:
        pytest.fail("Failed to create BM25 graph")

    bm25_indexer = BM25CodeIndexer(code_graph=graph)
    bm25_node_ids = {
        doc.metadata.get("node_id")
        for doc in bm25_indexer.documents
        if doc.metadata.get("node_id")
    }

    try:
        code_chunker = CodeChunker(language="python", max_lines_per_chunk=50)
        chunks = code_chunker.chunk_repository(str(repo_path))
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"Dense chunks creation failed: {exc}")

    dense_node_ids = {chunk.node_id for chunk in chunks if chunk.node_id}

    compatible_ids = dense_node_ids.intersection(bm25_node_ids)
    missing_ids = dense_node_ids - bm25_node_ids
    extra_bm25_ids = bm25_node_ids - dense_node_ids

    compatibility_rate = (
        len(compatible_ids) / len(dense_node_ids) * 100 if dense_node_ids else 0
    )

    assert not missing_ids, (
        "Found dense node IDs missing in BM25: "
        f"{sorted(missing_ids)} (compatibility={compatibility_rate:.1f}%)"
    )

    # Report extra BM25 IDs as context rather than failing the test. Some indices
    # may contain auxiliary nodes that dense chunking intentionally omits.
    if extra_bm25_ids:
        warnings.warn(
            f"BM25 contains node IDs without dense chunks: {sorted(extra_bm25_ids)}",
            UserWarning,
            stacklevel=1,
        )
