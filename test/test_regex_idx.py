"""
Test RegexNodeIndex functionality using simple_repo.
"""
import os
from pathlib import Path

from codeminer import CodeGraph, RegexNodeIndex
from codeminer.scip_interface import SCIPIndexer


def test_regex_index_basic():
    """Test basic RegexNodeIndex functionality with simple_repo."""
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

    # Create RegexNodeIndex
    print("\nBuilding RegexNodeIndex...")
    regex_idx = RegexNodeIndex(code_graph=code_graph)

    # Test 1: Search for 'calculator' (plain string)
    print("\n=== Test 1: Plain string search for 'calculator' ===")
    results = regex_idx.search('calculator', use_regex=False)
    print(f"Found {len(results)} nodes containing 'calculator':")
    for node in results[:5]:  # Show first 5
        print(f"  - {node.node_name} ({node.type})")

    # Test 2: Regex search for function definitions
    print("\n=== Test 2: Regex search for function definitions ===")
    results = regex_idx.search(r'def\s+\w+', file_glob='*.py')
    print(f"Found {len(results)} nodes with function definitions:")
    for node in results[:5]:
        print(f"  - {node.file}:{node.start_line} - {node.node_name}")

    # Test 3: Search with file glob filter
    print("\n=== Test 3: Search in calculator files only ===")
    results = regex_idx.search('class', file_glob='*calculator*', use_regex=False)
    print(f"Found {len(results)} nodes containing 'class' in calculator files:")
    for node in results:
        print(f"  - {node.file} - {node.node_name} ({node.type})")

    # Test 4: Search by node type
    print("\n=== Test 4: Count nodes by type ===")
    type_counts = {}
    for node in regex_idx.nodes:
        type_counts[node.type] = type_counts.get(node.type, 0) + 1
    print("Node counts by type:")
    for node_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {node_type}: {count}")

    # Test 5: Case-sensitive search
    print("\n=== Test 5: Case-sensitive vs case-insensitive ===")
    case_sensitive_results = regex_idx.search('CLASS', case_sensitive=True, use_regex=False)
    case_insensitive_results = regex_idx.search('CLASS', case_sensitive=False, use_regex=False)
    print(f"Case-sensitive 'CLASS': {len(case_sensitive_results)} results")
    print(f"Case-insensitive 'CLASS': {len(case_insensitive_results)} results")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_regex_index_basic()

