import argparse

from codeminer.env.process_swebench_data import (
    load_filter_swebench_dataset,
    process_swebench_instance,
)
from codeminer.search import CodeSearchEngine

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

        # use the CodeSearchEngine to search for code
        search_engine = CodeSearchEngine(
            repo_path=repo_path,
            llm_model=args_dict["model"],
            llm_temperature=0.0,
            top_k=5,
            language="english",
        )

        # Search with the problem statement
        results = search_engine.search_with_context(instance["problem_statement"])

        # Print the top results
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} (Score: {result['score']:.4f}) ---")
            print(f"File: {result.get('file')}")
            print(f"Symbol: {result.get('name')}")
            print(f"Lines: {result.get('start_line')}-{result.get('end_line')}")
            if "context_code" in result:
                print("\nCode Context:")
                print(result["context_code"])
