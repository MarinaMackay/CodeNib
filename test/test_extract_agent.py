import argparse

from codeminer.env.process_data import load_filter_swebench_dataset
from codeminer.extract_agent import KeywordExtractor

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

        # use the KeywordExtractor to extract keywords
        extractor = KeywordExtractor(
            model_name=args_dict["model"],
            config_path="../key.cfg",
            temperature=0.0,
        )
        result = extractor.extract_keywords(instance["problem_statement"])
        print(f"Extracted keywords: {result.keywords}")
        for keyword in result.keywords:
            print(f"Keyword: {keyword}")
