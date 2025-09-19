import os
import subprocess
from pathlib import Path

from codeminer.bm25_index import BM25CodeIndexer
from codeminer.scip_interface import SCIPIndexer


def setup_samplemod_repo():
    """Clone and set up the samplemod repository for testing."""
    # Define the test repo URL and path
    test_repo_url = "https://github.com/navdeep-G/samplemod.git"
    test_repo_path = Path("/tmp/samplemod-test")

    # Clone the repo if it doesn't exist
    if not test_repo_path.exists():
        print(f"Cloning sample module repository from {test_repo_url}...")
        subprocess.run(["git", "clone", test_repo_url, str(test_repo_path)], check=True)
    else:
        print(f"Using existing sample module repository at {test_repo_path}")

    return test_repo_path


def get_code_graph_from_samplemod(test_repo_path):
    """Create a code graph from the samplemod repository using SCIPIndexer."""
    # Create output file in the local directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    output_file = str(current_dir / "samplemod_index.json")

    # Create a new indexer for the samplemod repo
    repo_indexer = SCIPIndexer(test_repo_path)

    # Run the indexing pipeline
    graph = repo_indexer.run_pipeline(
        project_name="SampleModRepo",
        output_file=output_file,
        skip_index=False,
        skip_decode=False,
    )

    return graph


# Example usage
if __name__ == "__main__":
    # Get the samplemod repository
    repo_path = setup_samplemod_repo()

    # Try to use an existing JSON file for the graph if available
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    print("Creating code graph from samplemod repository...")
    graph = get_code_graph_from_samplemod(repo_path)

    # Print basic information about the graph
    if graph:
        print("\nGraph information:")
        graph.print_graph_basic_info()
    else:
        print("Failed to create code graph.")
        exit(1)
    print("\n--- BM25 Index Testing ---")

    # Create BM25 indexer with English stemming for method name matching
    indexer = BM25CodeIndexer(max_k=10, language="english")

    # Build index from code graph
    print("Building BM25 index from code graph...")
    indexer.build_index_from_graph(graph)

    # Test search queries with more comprehensive examples including variations and typos
    search_queries = [
        "hmm",  # Exact match
        "get_hmm",  # Partial match for "get_hmm_data"
        "gethmm",  # No underscore
        "get hmm",  # Different word segmentation
    ]
    # Save and load index example
    index_dir = current_dir / "bm25_index_test"
    print(f"\nSaving BM25 index to {index_dir}...")

    if not index_dir.exists():
        index_dir.mkdir(parents=True)

    indexer.save_index(str(index_dir))

    print("Loading BM25 index from disk...")
    new_indexer = BM25CodeIndexer()
    new_indexer.load_index(str(index_dir))

    # Test search with the loaded index
    print("\nSearching with loaded index:")

    print("\n--- Testing search with code content ---")
    content_query = search_queries[0]  # Use first query
    print(f"Searching for '{content_query}' with code content:")
    content_results = new_indexer.search(
        content_query, top_k=3, return_code_content=True, wrap_with_ln=True
    )

    for i, result in enumerate(content_results, 1):
        print(f"{i}. {result}")
        if result.content:
            print(f"\nContent preview:\n{result.content[:200]}...")
        print()

    print("\n--- BM25 Index Testing Completed ---")
    print("Test completed successfully.")
