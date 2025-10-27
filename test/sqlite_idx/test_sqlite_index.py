"""
Test SqliteNodeIndex functionality using simple_repo.
"""
import os
from pathlib import Path

from codeminer import CodeGraph, SqliteNodeIndex
from codeminer.scip_interface import SCIPIndexer


def test_sqlite_index_basic():
    """Test basic SqliteNodeIndex functionality with simple_repo."""
    # Setup paths
    simple_repo_path = Path(__file__).parent.parent / "simple_repo"
    output_path = Path.home() / ".codeminer" / "simple_repo_nodes_test"

    print(f"Testing with simple_repo at: {simple_repo_path}")
    print(f"Output path: {output_path}")

    # Build CodeGraph using SCIPIndexer
    indexer = SCIPIndexer(str(simple_repo_path), output_dir=str(output_path))
    code_graph = indexer.run_pipeline(
        project_name="simple_repo_test",
        skip_level="graph",  # Use cached graph if available
    )

    print(f"\nCodeGraph built with {code_graph.graph.vcount()} nodes")

    # Create SqliteNodeIndex
    print("\nBuilding SqliteNodeIndex...")
    sqlite_idx = SqliteNodeIndex(code_graph=code_graph)

    # Test 1: Get statistics
    print("\n=== Test 1: Statistics ===")
    stats = sqlite_idx.get_stats()
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Files indexed: {stats['files_indexed']}")
    print(f"Nodes by type: {stats['nodes_by_type']}")

    # Test 2: Query all nodes
    print("\n=== Test 2: Query all nodes (limit 5) ===")
    all_nodes = sqlite_idx.query("SELECT vertex_id, node_name, type FROM nodes LIMIT 5")
    for vid, name, ntype in all_nodes:
        print(f"  {vid}: {name} ({ntype})")

    # Test 3: GLOB pattern matching
    print("\n=== Test 4: GLOB pattern matching (calculator) ===")
    calc_nodes = sqlite_idx.query(
        "SELECT node_name, type FROM nodes WHERE node_name GLOB '*calculator*'"
    )
    print(f"Found {len(calc_nodes)} nodes matching '*calculator*':")
    for name, ntype in calc_nodes:
        print(f"  - {name} ({ntype})")

    # Test 4: Complex query
    print("\n=== Test 7: Complex query (count by type) ===")
    type_counts = sqlite_idx.query(
        "SELECT type, COUNT(*) as cnt FROM nodes GROUP BY type ORDER BY cnt DESC"
    )
    for ntype, cnt in type_counts:
        print(f"  {ntype}: {cnt}")

    

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_sqlite_index_basic()

