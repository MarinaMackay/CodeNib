#!/usr/bin/env python3
"""Test to check if all Dense chunk node_ids exist in BM25 graph nodes."""

from pathlib import Path

import pytest

from codeminer.code_chunker import CodeChunker
from codeminer.scip_interface import SCIPIndexer
from codeminer.sparse_idx.bm25_index import BM25CodeIndexer


@pytest.fixture
def test_setup():
    """Set up test environment"""
    repo_path = Path(__file__).parent / "simple_repo"
    output_path = Path.home() / ".codeminer" / "simple_repo_nodes_test"

    # Ensure the test repo exists
    if not repo_path.exists():
        pytest.skip(f"Test repository not found at {repo_path}")

    print(f"Testing with repository: {repo_path}")

    return {"repo_path": repo_path, "output_path": output_path}


def test_dense_compatibility(test_setup):
    """Test that all Dense chunk node_ids exist in BM25 graph nodes"""
    repo_path = test_setup["repo_path"]
    output_path = test_setup["output_path"]

    print("\n" + "=" * 80)
    print("DENSE COMPATIBILITY TEST")
    print("=" * 80)

    # === Create BM25 nodes ===
    print("\n1. Creating BM25 nodes...")
    repo_indexer = SCIPIndexer(repo_path, output_dir=output_path)
    graph = repo_indexer.run_pipeline(project_name="simple_repo_compat", force=True)

    assert graph, "Failed to create BM25 graph"

    bm25_indexer = BM25CodeIndexer(code_graph=graph)
    bm25_nodes = getattr(bm25_indexer, "nodes", [])

    # Extract all BM25 node IDs
    bm25_node_ids = {node_id for node_id in bm25_nodes if node_id}

    print(f"   Found {len(bm25_node_ids)} BM25 node IDs")
    print(f"   BM25 node IDs: {sorted(bm25_node_ids)}")

    # === Create Dense chunks ===
    print("\n2. Creating Dense chunks...")
    code_chunker = CodeChunker(language="python")
    chunks = code_chunker.chunk_repository(str(repo_path), languages=["python"])

    # Extract all Dense chunk node IDs
    dense_node_ids = set()
    for chunk in chunks:
        if chunk.node_id:  # Skip empty node_ids
            dense_node_ids.add(chunk.node_id)

    print(f"   Found {len(dense_node_ids)} Dense node IDs")
    print(f"   Dense node IDs: {sorted(dense_node_ids)}")

    # === Compatibility Check ===
    print("\n3. Compatibility Analysis:")
    print("-" * 50)

    # Check which Dense node IDs exist in BM25
    compatible_ids = dense_node_ids.intersection(bm25_node_ids)
    missing_ids = dense_node_ids - bm25_node_ids
    extra_bm25_ids = bm25_node_ids - dense_node_ids

    print(f"✅ Compatible node IDs ({len(compatible_ids)}):")
    for node_id in sorted(compatible_ids):
        print(f"   - {node_id}")

    if missing_ids:
        print(f"\n❌ Dense node IDs missing in BM25 ({len(missing_ids)}):")
        for node_id in sorted(missing_ids):
            print(f"   - {node_id}")
    else:
        print(f"\n✅ All Dense node IDs found in BM25!")

    if extra_bm25_ids:
        print(f"\n📝 BM25-only node IDs ({len(extra_bm25_ids)}):")
        for node_id in sorted(extra_bm25_ids):
            print(f"   - {node_id}")

    # === Results Summary ===
    print("\n" + "=" * 50)
    print("COMPATIBILITY SUMMARY")
    print("=" * 50)
    print(f"Dense chunks: {len(chunks)}")
    print(f"Dense node IDs: {len(dense_node_ids)}")
    print(f"BM25 node IDs: {len(bm25_node_ids)}")
    print(f"Compatible: {len(compatible_ids)}")
    print(f"Missing in BM25: {len(missing_ids)}")

    compatibility_rate = (
        len(compatible_ids) / len(dense_node_ids) * 100 if dense_node_ids else 0
    )
    print(f"Compatibility rate: {compatibility_rate:.1f}%")

    # Assert that all Dense node IDs exist in BM25
    assert (
        len(missing_ids) == 0
    ), f"Found {len(missing_ids)} Dense node IDs missing in BM25: {missing_ids}"

    print("\n✅ COMPATIBILITY TEST PASSED!")
    print("=" * 80)
