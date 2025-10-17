#!/usr/bin/env python3
"""
This script demonstrates the usage of SearchRerankPipeline on SWE-bench or LocBench datasets.

Usage:
    # Run on SWE-bench with default settings
    python examples/search_rerank.py --dataset swebench_lite
    
    # Run on LocBench with custom filter
    python examples/search_rerank.py --dataset locbench_v1 --filter-instance "^(joselc__life-sim-first-try-2)$"
    
    # Run on SWE-bench with custom embedding model
    python examples/search_rerank.py --dataset swebench_lite --embedding-model nomic-ai/CodeRankEmbed --embedding-provider huggingface
"""

import argparse
import sys
from pathlib import Path

from codeminer.model import SearchRerankPipeline
from codeminer.env.process_locbench_data import (
    load_filter_locbench_dataset,
    process_locbench_instance,
)
from codeminer.env.process_swebench_data import (
    load_filter_swebench_dataset,
    process_swebench_instance,
)
from codeminer.llm.llm_config import LLMProvider
from codeminer.log_utils import get_logger

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
        description="Run Search + Rerank Pipeline on benchmark datasets",
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
    
    # Embedding configuration
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nomic-ai/CodeRankEmbed",
        help="Embedding model name for dense retrieval",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default="huggingface",
        choices=["openai", "huggingface"],
        help="Embedding provider",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=768,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code for embedding model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for embedding encoding",
    )
    
    # Rerank configuration
    parser.add_argument(
        "--rerank-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B",
        help="Rerank model name",
    )
    parser.add_argument(
        "--rerank-provider",
        type=str,
        default="vllm_openai",
        choices=[pv.value for pv in LLMProvider],
        help="Rerank provider",
    )
    
    # Repository processing configuration
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["python"],
        help="Programming languages to process",
    )
    parser.add_argument(
        "--max-lines-per-chunk",
        type=int,
        default=100,
        help="Maximum lines per code chunk",
    )
    
    # Search configuration
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to return",
    )
    
    # Cache configuration
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=Path.home() / ".codeminer",
        help="Cache directory for vector store (default: ~/.codeminer)",
    )
    
    return parser.parse_args()


def run_pipeline(args):
    """Run the search + rerank pipeline on the specified dataset."""
    
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
        # Process instance to get repo path
        repo_path = dataset_config["processor"](instance)
        
        # Initialize pipeline
        embedding_model_kwargs = {
            "trust_remote_code": args.trust_remote_code,
            "encode_kwargs": {
                "batch_size": args.batch_size,
            },
        }
        
        pipeline = SearchRerankPipeline(
            repo_path=repo_path,
            embedding_model=args.embedding_model,
            embedding_provider=args.embedding_provider,
            embedding_dimension=args.embedding_dimension,
            embedding_model_kwargs=embedding_model_kwargs,
            rerank_model=args.rerank_model,
            rerank_provider=LLMProvider(args.rerank_provider),
            languages=args.languages,
            max_lines_per_chunk=args.max_lines_per_chunk,
            cache_dir=args.cache_dir,
        )
        
        # Query the pipeline
        query = instance["problem_statement"]
        results = pipeline.query(query=query, top_k=args.top_k)
        
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
    logger.info(f"Embedding model: {args.embedding_model}")
    logger.info(f"Rerank model: {args.rerank_model}")
    logger.info(f"Top-K: {args.top_k}")
    
    run_pipeline(args)


if __name__ == "__main__":
    main()
