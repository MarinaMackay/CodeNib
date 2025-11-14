#!/usr/bin/env python3
"""
This script demonstrates the usage of RetrieveRerankPipeline on SWE-bench or LocBench datasets.
Before running the pipeline, start a vLLM server for the rerank model:

```bash
python scripts/start_vllm_server.py --model Qwen/Qwen2.5-Coder-7B
```

Usage:
    # Run on SWE-bench with default settings
    python examples/retrieve_rerank.py --dataset swebench_lite

    # Run on LocBench with custom filter
    python examples/retrieve_rerank.py --dataset locbench_v1 --filter-instance "^(joselc__life-sim-first-try-2)$"

    # Run on SWE-bench with custom embedding model
    python examples/retrieve_rerank.py --dataset swebench_lite --embedding-model nomic-ai/CodeRankEmbed --embedding-provider huggingface

    # Run with hybrid (dense + sparse) retrieval before rerank
    python examples/retrieve_rerank.py --dataset swebench_lite --retrieval-mode hybrid

    # Override cache directories (one for indices, one for repos)
    python examples/retrieve_rerank.py --dataset swebench_lite --index-cache-dir /tmp/codeminer/index --repo-cache-dir ~/.codeminer/
"""

import argparse
import json
import os
from pathlib import Path

from codeminer.env.process_locbench_data import (
    load_filter_locbench_dataset,
    process_locbench_instance,
)
from codeminer.env.process_swebench_data import (
    load_filter_swebench_dataset,
    process_swebench_instance,
)
from codeminer.eval.retrieval_eval import (
    aggregate_metrics,
    average_metrics,
    collect_targets,
    evaluate_predictions,
)
from codeminer.llm.llm_config import LLMProvider
from codeminer.log_utils import get_logger
from codeminer.model import RetrieveRerankPipeline, build_retrieve_plan

logger = get_logger(__name__)

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
        description="Run Retrieve + Rerank Pipeline on benchmark datasets",
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
        default=".*",
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
        default=32,
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
    parser.add_argument(
        "--rerank-window-size",
        type=int,
        default=10,
        help=(
            "Number of candidates per LLM rerank window (None processes all candidates at once)."
        ),
    )
    parser.add_argument(
        "--rerank-window-step",
        type=int,
        default=None,
        help=(
            "Stride between rerank windows; defaults to the window size when unspecified."
        ),
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
        default=300,
        help="Maximum lines per code chunk",
    )

    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default="dense",
        choices=["dense", "sparse", "hybrid"],
        help="Retrieval plan to run (dense-only, BM25-only, or hybrid).",
    )

    # Evaluation configuration
    default_eval_path = Path.home() / ".codeminer" / "swebench_verified_gt.json"
    parser.add_argument(
        "--eval-instances",
        type=str,
        default=str(default_eval_path),
        help=(
            "Path to JSON file containing evaluation annotations (target_files, symbols_*). "
            "Defaults to ~/.codeminer/swebench_verified_gt.json."
        ),
    )
    parser.add_argument(
        "--metrics-k",
        type=int,
        nargs="+",
        default=[10],
        help="Cutoffs for accuracy/precision/recall reporting",
    )

    # Retrieval configuration
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to return",
    )

    # Cache configuration
    parser.add_argument(
        "--index-cache-dir",
        type=str,
        default="/mnt/data/codeminer",
        help="Directory to store embedding/vector indices",
    )
    parser.add_argument(
        "--repo-cache-dir",
        type=str,
        default="~/.codeminer/",
        help="Directory to cache cloned repositories for dataset instances",
    )

    return parser.parse_args()


def load_eval_metadata(path: str):
    resolved = Path(os.path.expanduser(path)).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Evaluation annotations file not found at {resolved}. "
            "Generate it via scripts/swebench_gt_locate.py or point --eval-instances elsewhere."
        )
    with open(resolved, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict) and "instances" in payload:
        records = payload["instances"]
    elif isinstance(payload, list):
        records = payload
    else:
        records = [payload]
    metadata = {}
    for entry in records:
        instance_id = entry.get("instance_id")
        if instance_id:
            metadata[instance_id] = entry
    return metadata


def run_pipeline(args):
    """Run the retrieve + rerank pipeline on the specified dataset."""

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

    eval_metadata = load_eval_metadata(args.eval_instances)
    retrieve_plan = build_retrieve_plan(args.retrieval_mode)
    aggregate = {}
    metrics_k = sorted(set(args.metrics_k))
    eval_count = 0

    # Process each instance
    for _, instance in enumerate(dataset_instances):
        # Process instance to get repo path
        repo_path = dataset_config["processor"](instance, cache_dir=args.repo_cache_dir)

        # Compute index path
        instance_id = instance["instance_id"]
        instance_dir_name = instance_id.replace("/", "__")
        index_path = Path(args.index_cache_dir) / instance_dir_name

        # Initialize pipeline
        embedding_model_kwargs = {
            "trust_remote_code": args.trust_remote_code,
            "encode_kwargs": {
                "batch_size": args.batch_size,
            },
        }

        pipeline = RetrieveRerankPipeline(
            repo_path=repo_path,
            index_path=str(index_path),
            embedding_model=args.embedding_model,
            embedding_provider=args.embedding_provider,
            embedding_dimension=args.embedding_dimension,
            embedding_model_kwargs=embedding_model_kwargs,
            rerank_model=args.rerank_model,
            rerank_provider=LLMProvider(args.rerank_provider),
            languages=args.languages,
            max_lines_per_chunk=args.max_lines_per_chunk,
            retrieval_plan=retrieve_plan,
            rerank_window_size=args.rerank_window_size,
            rerank_window_step=args.rerank_window_step,
        )

        # Query the pipeline
        query = instance["problem_statement"]
        results = pipeline.query(query=query, top_k=max(max(metrics_k), args.top_k))

        # for i, node in enumerate(results[: args.top_k]):
        #     logger.info("--------------------------------")
        #     logger.info(f"Rank {i + 1} (Score: {node.score:.4f})")
        #     logger.info(f"  Node Name: {node.node_name}")
        #     logger.info(f"  Node Type: {node.type}")
        #     logger.info(f"  File: {node.file}")
        #     logger.info(f"  Lines: {node.start_line}-{node.end_line}")
        #     logger.info(f"  Content: {node.content}")

        metadata = eval_metadata.get(instance_id)
        if metadata:
            target_files, target_symbols = collect_targets(metadata)
            metrics = evaluate_predictions(
                nodes=results,
                target_files=target_files,
                target_symbols=target_symbols,
                ks=metrics_k,
            )
            aggregate_metrics(aggregate, metrics)
            eval_count += 1
            logger.info("Evaluation metrics for %s:", instance_id)
            for scope, per_k in metrics.items():
                for k, stats in per_k.items():
                    logger.info(
                        "  [%s] k=%d acc=%.3f prec=%.3f recall=%.3f hits=%d",
                        scope,
                        k,
                        stats["accuracy"],
                        stats["precision"],
                        stats["recall"],
                        int(stats["hits"]),
                    )

    if aggregate and eval_count:
        averaged = average_metrics(aggregate, eval_count)
        logger.info(
            "=== Aggregate Retrieval Metrics (over %d instances) ===", eval_count
        )
        for scope, per_k in averaged.items():
            for k, stats in per_k.items():
                logger.info(
                    "[%s] k=%d acc=%.3f prec=%.3f recall=%.3f avg_hits=%.3f",
                    scope,
                    k,
                    stats["accuracy"],
                    stats["precision"],
                    stats["recall"],
                    stats["avg_hits"],
                )


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
