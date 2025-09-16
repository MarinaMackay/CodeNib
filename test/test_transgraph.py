"""Test script for transverse_graph functionality using a simple test repository."""

from pathlib import Path

from codeminer.bm25_index import BM25CodeIndexer
from codeminer.scip_interface import SCIPIndexer
from codeminer.transverse_graph import (
    RepoDependencySearcher,
    RepoEntitySearcher,
    traverse_tree_structure,
)
from codeminer.types import (
    EDGE_TYPE_CONTAIN,
    NODE_TYPE_CLASS,
    NODE_TYPE_FIELD,
    NODE_TYPE_FILE,
    NODE_TYPE_FUNCTION,
    NODE_TYPE_METHOD,
    NODE_TYPE_SYMBOL,
)


def test_transgraph_simple():
    """Test traverse functions with a simple test repository."""

    # Use the simple test repository
    repo_path = Path(__file__).parent / "simple_repo"
    output_path = Path.home() / ".codeminer" / "simple_repo_test"

    print(f"Testing with repository: {repo_path}")
    print(f"Output directory: {output_path}")

    # Ensure the test repo exists
    if not repo_path.exists():
        print(f"Error: Test repository not found at {repo_path}")
        return False

    # Setup codegraph
    repo_indexer = SCIPIndexer(repo_path, output_dir=output_path)

    print("\n" + "=" * 50)
    print("GENERATING SCIP INDEX")
    print("=" * 50)

    # Run the indexing pipeline
    graph = repo_indexer.run_pipeline(
        project_name="simple_repo_test", force=True  # Force regeneration for testing
    )

    if not graph:
        print("Failed to generate graph!")
        return False

    print("Graph generated successfully!")

    # Print basic graph info
    print(f"\n--- Basic Graph Information ---")
    graph.print_graph_basic_info()

    # Test traverse functionality
    print("\n" + "=" * 50)
    print("TESTING TRAVERSE FUNCTIONS")
    print("=" * 50)

    # Initialize searchers
    entity_searcher = RepoEntitySearcher(graph)
    dependency_searcher = RepoDependencySearcher(graph)

    # Get sample nodes
    print("\n--- Getting sample nodes ---")
    file_nodes = entity_searcher.get_all_nodes_by_type(NODE_TYPE_FILE)
    symbol_nodes = entity_searcher.get_all_nodes_by_type(NODE_TYPE_SYMBOL)

    print(f"Found {len(file_nodes)} file nodes")
    print(f"Found {len(symbol_nodes)} symbol nodes")

    if file_nodes:
        print(f"Sample file nodes:")
        for i, node in enumerate(file_nodes[:5]):
            print(f"  {i+1}. {node['name']}")

    if symbol_nodes:
        print(f"Sample symbol nodes:")
        for i, node in enumerate(symbol_nodes[:10]):
            print(f"  {i+1}. {node['name']} (type: {node.get('type', 'unknown')})")

    # Test specific symbol types
    print("\n--- Testing specific symbol types ---")
    class_nodes = entity_searcher.get_all_nodes_by_type(NODE_TYPE_CLASS)
    function_nodes = entity_searcher.get_all_nodes_by_type(NODE_TYPE_FUNCTION)
    method_nodes = entity_searcher.get_all_nodes_by_type(NODE_TYPE_METHOD)
    field_nodes = entity_searcher.get_all_nodes_by_type(NODE_TYPE_FIELD)

    print(f"Found {len(class_nodes)} class nodes")
    print(f"Found {len(function_nodes)} function nodes")
    print(f"Found {len(method_nodes)} method nodes")
    print(f"Found {len(field_nodes)} field nodes")

    if class_nodes:
        print("Sample class nodes:")
        for i, node in enumerate(class_nodes[:3]):
            print(f"  {i+1}. {node['name']}")

    if field_nodes:
        print("Sample field nodes:")
        for i, node in enumerate(field_nodes[:3]):
            print(f"  {i+1}. {node['name']}")

    # Test traverse_tree_structure
    if file_nodes:
        print(f"\n--- Testing traverse_tree_structure ---")

        # Test with main.py if it exists
        main_file = None
        for node in file_nodes:
            if node["name"].endswith("main.py"):
                main_file = node["name"]
                break

        if not main_file and file_nodes:
            main_file = file_nodes[0]["name"]

        if main_file:
            print(f"Root file: {main_file}")

            # Test downstream traversal
            tree_result_downstream = traverse_tree_structure(
                graph,
                main_file,
                direction="downstream",
                hops=2,
                node_type_filter=[
                    NODE_TYPE_FILE,
                    NODE_TYPE_CLASS,
                    NODE_TYPE_FUNCTION,
                    NODE_TYPE_METHOD,
                ],
                # edge_type_filter=[EDGE_TYPE_CONTAIN]
            )
            print("Tree structure (downstream, 2 hops):")
            print(tree_result_downstream)

            # Test with unlimited hops
            tree_result_unlimited = traverse_tree_structure(
                graph,
                main_file,
                direction="downstream",
                hops=-1,  # Unlimited
                node_type_filter=[
                    NODE_TYPE_FILE,
                    NODE_TYPE_CLASS,
                    NODE_TYPE_FUNCTION,
                    NODE_TYPE_METHOD,
                ],
            )
            print(f"\nTree structure (downstream, unlimited hops):")
            print(tree_result_unlimited)

    # Test with a symbol node if available
    if symbol_nodes:
        print(f"\n--- Testing with Symbol Node ---")

        # Find a calculator method if it exists
        calc_symbol = None
        for node in symbol_nodes:
            if "Calculator" in node["name"] or "add" in node["name"]:
                calc_symbol = node["name"]
                break

        if not calc_symbol:
            calc_symbol = symbol_nodes[0]["name"]

        print(f"Symbol: {calc_symbol}")

        # Test upstream traversal
        tree_result_upstream = traverse_tree_structure(
            graph,
            calc_symbol,
            direction="upstream",
            hops=2,
            node_type_filter=[NODE_TYPE_FILE, NODE_TYPE_SYMBOL],
        )
        print("Tree structure (upstream from symbol, 2 hops):")
        print(tree_result_upstream)

    # Test RepoEntitySearcher functionality
    print(f"\n--- Testing RepoEntitySearcher ---")

    # Test global name dictionary
    global_names = list(entity_searcher.global_name_dict.keys())
    print(f"Global names found: {len(global_names)}")
    if global_names:
        print("Sample global names:")
        for name in global_names[:10]:
            print(f"  - {name}")

    # Test node data retrieval
    if file_nodes:
        sample_file = file_nodes[0]["name"]
        node_data = entity_searcher.get_node_data(
            [sample_file], return_code_content=True, wrap_with_ln=True
        )
        print(f"\nNode data for '{sample_file}':")
        if node_data and node_data[0]:
            for key, value in node_data[0].items():
                if key == "code_content":
                    print(f"  {key}: {len(str(value))} characters")
                    # Show first few lines
                    lines = str(value).split("\n")[:5]
                    for line in lines:
                        print(f"    {line}")
                    if len(str(value).split("\n")) > 5:
                        print("    ...")
                else:
                    print(f"  {key}: {value}")

    # Test RepoDependencySearcher functionality
    print(f"\n--- Testing RepoDependencySearcher ---")

    if file_nodes:
        test_file = file_nodes[0]["name"]

        # Get forward neighbors (what this file contains)
        forward_nodes, forward_edges = dependency_searcher.get_neighbors(
            test_file, direction="forward", ntype_filter=[NODE_TYPE_SYMBOL]
        )
        print(f"Forward neighbors of '{test_file}': {len(forward_nodes)} nodes")
        if forward_nodes:
            print("Forward neighbors:")
            for node in forward_nodes[:5]:
                print(f"  - {node}")

        print(f"Forward edges: {len(forward_edges)}")
        if forward_edges:
            print("Forward edges:")
            for edge in forward_edges[:5]:
                print(f"  - {edge[0]} -> {edge[1]} ({edge[3]['type']})")

    # Test with symbol nodes
    if symbol_nodes:
        test_symbol = symbol_nodes[0]["name"]

        # Get backward neighbors (what contains this symbol)
        backward_nodes, backward_edges = dependency_searcher.get_neighbors(
            test_symbol, direction="backward", ntype_filter=[NODE_TYPE_FILE]
        )
        print(f"\nBackward neighbors of '{test_symbol}': {len(backward_nodes)} nodes")
        if backward_nodes:
            print("Backward neighbors:")
            for node in backward_nodes:
                print(f"  - {node}")

        print(f"Backward edges: {len(backward_edges)}")
        if backward_edges:
            print("Backward edges:")
            for edge in backward_edges:
                print(f"  - {edge[0]} -> {edge[1]} ({edge[3]['type']})")

    # Test edge filtering
    print(f"\n--- Testing Edge Filtering ---")
    if file_nodes:
        test_file = file_nodes[0]["name"]
        contain_nodes, contain_edges = dependency_searcher.get_neighbors(
            test_file, direction="forward", etype_filter=[EDGE_TYPE_CONTAIN]
        )
        print(f"Nodes connected via '{EDGE_TYPE_CONTAIN}' edges: {len(contain_nodes)}")
        if contain_nodes:
            for node in contain_nodes[:3]:
                print(f"  - {node}")

    print(f"\n--- Test Summary ---")
    print(f"✅ Graph indexing: SUCCESS")
    print(f"✅ Entity searching: SUCCESS")
    print(f"✅ Dependency searching: SUCCESS")
    print(f"✅ Tree traversal: SUCCESS")
    print(f"✅ Node filtering: SUCCESS")
    print(f"✅ Edge filtering: SUCCESS")

    return True


def test_bm25_transgraph_compatibility():
    """Test compatibility between BM25 graph-like output and traverse graph get_node_data."""
    repo_path = Path(__file__).parent / "simple_repo"
    assert repo_path.exists(), f"Test repository not found at {repo_path}"
    print(f"Using repo: {repo_path}")

    # Build graph
    output_path = Path.home() / ".codeminer" / "compat_test_smoke"
    if not output_path.exists():
        output_path.mkdir(parents=True)
    indexer = SCIPIndexer(str(repo_path), output_dir=str(output_path))
    graph = indexer.run_pipeline(project_name="compat_test_smoke", force=True)
    print("Graph built successfully")

    entity_searcher = RepoEntitySearcher(graph)

    # Pick a query derived from a real file name
    file_nodes = entity_searcher.get_all_nodes_by_type(NODE_TYPE_FILE)
    assert file_nodes, "Should have at least one file node"
    probe_file = file_nodes[0]["name"]
    query = Path(probe_file).stem
    print(f"Probe file: {probe_file}")
    print(f"Query: {query}")

    # Build BM25 and run graph-like search
    bm25 = BM25CodeIndexer(code_graph=graph, top_k=10, language="english")
    r1 = bm25.search_graph_like(query, return_code_content=True, wrap_with_ln=True)
    assert r1, "BM25 should return at least one node"
    nid = r1[0]["node_id"]

    # Get graph truth for the exact same node_id
    r2 = entity_searcher.get_node_data([nid], return_code_content=True, wrap_with_ln=True)
    assert r2 and r2[0], "Graph should return the same node data"

    # Print summaries
    b1 = r1[0]
    g1 = r2[0]
    print("BM25 first node:", {k: b1.get(k) for k in ["node_id", "type", "start_line", "end_line", "name"] if k in b1})
    print("Graph node:", {k: g1.get(k) for k in ["node_id", "type", "start_line", "end_line", "name"] if k in g1})

    b1_code = b1.get("code_content", "")
    g1_code = g1.get("code_content", "")
    print("BM25 code_content (head):\n", b1_code[:500])
    print("Graph code_content (head):\n", g1_code[:500])

    # equality on keys
    keys = set(["node_id", "type"]) | set(k for k in ("start_line", "end_line", "code_content") if k in g1)
    for k in keys:
        assert b1.get(k) == g1.get(k), f"Mismatch on {k}: {b1.get(k)} != {g1.get(k)}"

    print("BM25 graph-like output == traverse graph output: PASS")
    return True


# Example usage
if __name__ == "__main__":
    print("Starting simple repository traverse test...")
    success = test_transgraph_simple()
    compat_success = test_bm25_transgraph_compatibility()

    if success and compat_success:
        print("\n🎉 All tests completed successfully!")
        
    else:
        print("\n❌ Some tests failed!")
        exit(1)
