import argparse
from pathlib import Path

from codeminer.agent.rerank_agent import RerankAgent
from codeminer.env.process_data import (
    load_filter_swebench_dataset,
    process_swebench_instance,
)
from codeminer.graph.roi_subgraph import ROISubgraph
from codeminer.llm.llm_config import LLMConfig, LLMProvider
from codeminer.scip_interface import SCIPIndexer
from codeminer.sparse_idx.bm25_index import BM25CodeIndexer

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
        indexer = BM25CodeIndexer(max_k=10, language="english")

        # Build the index from the code graph
        indexer.build_index_from_graph(graph)
        # Search for the node name
        print(f"Searching for node name: {node_name}")
        results = indexer.search(node_name, top_k=5)
        print(f"Search results: {results}")
        # Extract node names from search results
        node_names = [result.node_name for result in results]
        print(f"Node names: {node_names}")
        # Create ROISubgraph object
        roi_subgraph = ROISubgraph(graph)
        # Extract subgraph with k-hop neighbors
        k_hop = 2
        subgraph = roi_subgraph.extract_subgraph(node_names, k_hop)
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

        # Use rerank agent to rank the filtered nodes by relevance to the problem statement
        print(
            f"\nReranking {len(filtered_nodes)} nodes by relevance to problem statement..."
        )
        llm_config = LLMConfig(
            model_name="gpt-4o",
            provider=LLMProvider.OPENAI,
        )
        rerank_agent = RerankAgent(llm_config=llm_config)
        ranked_nodes = rerank_agent.rerank_nodes(
            query=instance["problem_statement"], nodes=filtered_nodes, top_k=10
        )

        # Print top ranked nodes with scores
        print(f"\nTop ranked nodes (showing top 5):")
        for i, node in enumerate(ranked_nodes[:5]):
            print(f"Rank {i+1} (Score: {node.score:.3f})")
            print(f"  Node Name: {node.node_name}")
            print(f"  Node Type: {node.type}")
            print(f"  Node File: {node.file}")
            print(f"  Lines: {node.start_line}-{node.end_line}")
            print()

        # Also demonstrate the rerank_with_metadata method for detailed results
        print("Detailed ranking with metadata:")
        detailed_results = rerank_agent.rerank_with_metadata(
            query=instance["problem_statement"],
            nodes=filtered_nodes,
            top_k=3,
            include_content=True,
        )

        for result in detailed_results:
            print(f"Rank {result['rank']} (Score: {result['score']:.3f})")
            print(f"  Name: {result['node_name']}")
            print(f"  Type: {result['type']}")
            print(f"  File: {result['file']}")
            print(f"  Preview: {result.get('content_preview', 'N/A')}")
            print()
