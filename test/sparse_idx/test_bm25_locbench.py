import argparse
from pathlib import Path

from codeminer.env import load_filter_locbench_dataset, process_locbench_instance
from codeminer.scip_interface import SCIPIndexer
from codeminer.sparse_idx.bm25_index import BM25CodeIndexer

args_dict = {
    "model": "gpt-4o",
    "dataset": "czlll/Loc-Bench_V1",
    "split": "test",
    "filter_instance": "^(sympy__sympy-27223)$",
}


def test_bm25_index():
    """Test compatibility between BM25 graph-like output and traverse graph get_node_data."""
    args = argparse.Namespace(**args_dict)
    dataset = load_filter_locbench_dataset(args=args)
    for _, instance in enumerate(dataset):
        print(
            f"Loaded instance: {instance['instance_id']} from repo {instance['repo']}"
        )
        print(f"Base commit: {instance['base_commit']}")
        print(f"Problem statement: {instance['problem_statement']}")
        repo_path = process_locbench_instance(instance)
        # set output path with ~/.codeminer/instance_id
        output_path = str(Path.home()) + "/.codeminer/" + instance["instance_id"]

        # setup codegraph
        repo_indexer = SCIPIndexer(repo_path, output_dir=output_path)

        # Run the indexing pipeline, allowing skip_index and skip_decode for faster tests
        graph = repo_indexer.run_pipeline(
            project_name="test_swebench",
        )

        # setup bm25 indexer
        bm25_indexer = BM25CodeIndexer()
        bm25_indexer.build_index_from_graph(graph)

        node_check = "sympy/utilities/lambdify.py::lambdify"
        # Check that specific node is in the indexed nodes
        assert any(
            node == node_check for node in bm25_indexer.nodes
        ), f"Node {node_check} not found in indexed nodes"

        query = "lambdify"
        # query_test = "separability_matrix()."
        # query = query_test
        results_filtered = bm25_indexer.search(
            query, top_k=10, return_code_content=False, filter_test=True
        )
        print(f"BM25 query results for '{query}':")
        for result in results_filtered:
            print(f"Node: {result.node_name}")
            print(f"  {result}")
            # Verify no test files in results
            assert not any(
                word.startswith("test")
                for word in result.node_name.lower()
                .replace("_", " ")
                .replace("/", " ")
                .split()
            )
