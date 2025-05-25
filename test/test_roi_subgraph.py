import argparse
from pathlib import Path

from codeminer.api import SimilarityAPI
from codeminer.bm25_index import BM25CodeIndexer
from codeminer.env.process_data import (
    load_filter_swebench_dataset,
    process_swebench_instance,
)
from codeminer.roi_subgraph import ROISubgraph
from codeminer.scip_interface import SCIPIndexer

args_dict = {
    "model": "gpt-4o",
    "dataset": "princeton-nlp/SWE-bench_Lite",
    "split": "test",
    "filter_instance": "^(astropy__astropy-12907)$",
}


# Example usage
if __name__ == "__main__":
    # load instance from command line
    args = argparse.Namespace(**args_dict)
    dataset = load_filter_swebench_dataset(args=args)
    for _, instance in enumerate(dataset):
        print(
            f"Loaded instance: {instance['instance_id']} from repo {instance['repo']}"
        )
        print(f"Base commit: {instance['base_commit']}")
        print(f"Problem statement: {instance['problem_statement']}")
        repo_path = process_swebench_instance(instance)
        # set output path with ~/.codeminer/instance_id
        output_path = str(Path.home()) + "/.codeminer/" + instance["instance_id"]

        # setup codegraph
        repo_indexer = SCIPIndexer(repo_path, output_dir=output_path)

        # Run the indexing pipeline, allowing skip_index and skip_decode for faster tests
        graph = repo_indexer.run_pipeline(
            project_name="test_swebench",
        )

        # get node info
        node_name = "astropy.modeling.separable/separability_matrix()."

        # Create BM25 indexer with English stemming for method name matching
        indexer = BM25CodeIndexer(top_k=5, language="english")

        # Build the index from the code graph
        indexer.build_index_from_graph(graph)
        # Search for the node name
        print(f"Searching for node name: {node_name}")
        results = indexer.search(node_name)
        print(f"Search results: {results}")
        # Extract node IDs from search results
        node_ids = [result["node_id"] for result in results]
        print(f"Node IDs: {node_ids}")
        # Create ROISubgraph object
        roi_subgraph = ROISubgraph(graph)
        # Extract subgraph with k-hop neighbors
        k_hop = 2
        subgraph = roi_subgraph.extract_subgraph(node_ids, k_hop)
        # get filtered subgraph nodes
        filtered_nodes = roi_subgraph.get_filtered_subgraph_nodes(subgraph)

        # Print filtered nodes (sample 3 nodes), return type is NodeWithContent
        for i, node in enumerate(filtered_nodes[:3]):
            print(f"Filtered node {i+1}")
            print(f"Node Name: {node.node_name}")
            print(f"Node Type: {node.type}")
            print(f"Node Content: {node.content}")
            print(f"Node File: {node.file}")
            print(f"Node Start Line: {node.start_line}")
            print(f"Node End Line: {node.end_line}")

        # use SimilarityAPI to query the filtered nodes's content
        for i, node in enumerate(filtered_nodes):
            print(f"Querying node {i+1} content")
            # query the node content
            result = SimilarityAPI.query(node.content, instance["problem_statement"])
            print(f"Node Name: {node.node_name}, Query Result: {result}")
            # Print node attributes
