#!/usr/bin/env python3
"""
This script demonstrates the usage of AgentlessPipeline on SWE-bench or LocBench datasets.
Before running the pipeline, start a vLLM server for the llm model:

```bash
python scripts/start_vllm_server.py --model Qwen/Qwen2.5-Coder-7B
```

Usage example:
    python examples/agentless.py --dataset swebench_lite  --filter-instance "^(psf__requests-1963)$" --llm-model "Qwen/Qwen3-32B" --llm-provider "vllm_openai" 
"""

import argparse
import sys
from pathlib import Path

from codeminer.model import AgentlessPipeline
from codeminer.env.process_locbench_data import (
    load_filter_locbench_dataset,
    process_locbench_instance,
)
from codeminer.env.process_swebench_data import (
    load_filter_swebench_dataset,
    process_swebench_instance,
)
from codeminer.log_utils import get_logger
from codeminer.llm.llm_config import LLMProvider

logger = get_logger(__name__)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


DATASET_CONFIGS = {
    "swebench_lite": {
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "split": "test",
        "loader": load_filter_swebench_dataset,
        "processor": process_swebench_instance,
    },
    "locbench_v1": {
        "dataset": "czlll/Loc-Bench_V1",
        "split": "test",
        "loader": load_filter_locbench_dataset,
        "processor": process_locbench_instance,
    },
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Agentless Pipeline on benchmark datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["swebench_lite", "locbench_v1"],
        help="Type of dataset to run on",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--filter-instance",
        type=str,
        default=None,
        help="Regex pattern to filter instances (None processes all instances)",
    )
    
    # LLM configuration
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o",
        help="LLM model name for localization",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="openai",
        choices=[e.value for e in LLMProvider],
        help="LLM provider",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="Temperature for LLM sampling",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=2048,
        help="Max tokens for LLM response",
    )
    
    # Localization configuration
    parser.add_argument(
        "--top-n-files",
        type=int,
        default=3,
        help="Number of files to localize in Stage 1",
    )
    
    # Repository processing configuration
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["python"],
        help="Programming languages to process",
    )
    
    # Cache configuration
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=Path.home() / ".codeminer",
        help="Cache directory for code graph (default: ~/.codeminer)",
    )
    
    return parser.parse_args()


def run_pipeline(args):
    """Run the Agentless pipeline on the specified dataset."""
    
    # Get dataset configuration
    dataset = args.dataset
    dataset_config = DATASET_CONFIGS[dataset]
    
    # Prepare dataset args
    dataset_args = argparse.Namespace(
        dataset=dataset_config["dataset"],
        split=args.split,
        filter_instance=args.filter_instance,
    )
    
    # Load dataset
    dataset_instances = dataset_config["loader"](args=dataset_args)
    
    if len(dataset_instances) == 0:
        raise ValueError(f"No instances found in {dataset} dataset")
    
    logger.info(f"Loaded {len(dataset_instances)} instance(s)")
    
    # Process each instance
    for _, instance in enumerate(dataset_instances):
        # Process instance to get repo path and commit
        repo_info = dataset_config["processor"](instance)
        
        # Extract repo_path and repo_commit
        if isinstance(repo_info, dict):
            repo_path = repo_info.get("repo_path")
            repo_commit = repo_info.get("base_commit") or repo_info.get("commit", "HEAD")
        else:
            repo_path = repo_info
            repo_commit = instance.get("base_commit", "HEAD")
        
        # Initialize pipeline
        pipeline = AgentlessPipeline(
            repo_path=repo_path,
            repo_commit=repo_commit,
            llm_model=args.llm_model,
            llm_provider=args.llm_provider,
            llm_temperature=args.llm_temperature,
            llm_max_tokens=args.llm_max_tokens,
            top_n_files=args.top_n_files,
            languages=args.languages,
            cache_dir=args.cache_dir,
        )
        
        # Query the pipeline
        query = instance["problem_statement"]
        results = pipeline.query(problem_statement=query)
        
        for i, node in enumerate(results):
            #TODO: save the results to a file
            logger.info(f"Rank {i + 1} (Score: {node.score:.4f})")
            logger.info(f"  Node Name: {node.node_name}")
            logger.info(f"  Node Type: {node.type}")
            logger.info(f"  File: {node.file}")
            logger.info(f"  Lines: {node.start_line}-{node.end_line}")


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info(f"Dataset type: {args.dataset}")
    logger.info(f"LLM model: {args.llm_model}")
    logger.info(f"LLM provider: {args.llm_provider}")
    logger.info(f"Top-N files: {args.top_n_files}")
    
    run_pipeline(args)


if __name__ == "__main__":
    main()

