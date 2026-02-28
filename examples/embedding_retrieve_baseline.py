#!/usr/bin/env python3
"""
Embedding-only retrieval baseline script.

This script retrieves top-K nodes using embedding similarity search over the
entire codebase, for comparison with the graph-based baseline.

Usage:
    python examples/embedding_retrieve_baseline.py --dataset swebench_lite

    # Single instance
    python examples/embedding_retrieve_baseline.py --dataset swebench_lite \\
        --filter-instance "^(astropy__astropy-12907)$"
"""
import argparse
import json
import time
from pathlib import Path

from codeminer.dataset.locbench import LocbenchDataset
from codeminer.dataset.swebench import SwebenchDataset
from codeminer.eval.retrieval_eval import (
    aggregate_metrics,
    average_metrics,
    collect_targets,
    evaluate_predictions,
    extract_predictions,
)
from codeminer.index.embedding import build_hierarchical_vector_store
from codeminer.log_utils import get_logger
from codeminer.types import QueriedNode

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embedding-only retrieval baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["swebench_lite", "locbench_v1"],
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--filter-instance", type=str, default=".*")
    parser.add_argument("--topk", type=int, default=50)

    # Embedding config
    parser.add_argument(
        "--embedding-model", type=str, default="nomic-ai/CodeRankEmbed",
    )
    parser.add_argument(
        "--embedding-provider", type=str, default="huggingface",
    )
    parser.add_argument(
        "--embedding-dimension", type=int, default=768,
    )

    # Evaluation
    parser.add_argument(
        "--eval-instances", type=str, default=None,
        help="Path to eval annotations JSON. Auto-generated if not provided.",
    )
    parser.add_argument(
        "--metrics-k", type=int, nargs="+", default=[1, 3, 5, 10, 15, 20],
    )

    # Cache
    parser.add_argument(
        "--index-cache-dir", type=str, default="/mnt/data/codeminer",
    )
    parser.add_argument(
        "--repo-cache-dir", type=str, default="~/.codeminer/",
    )
    parser.add_argument("--result-path", type=str, default=None)

    return parser.parse_args()


def run_embedding_pipeline(args):
    """Run the embedding-only retrieval baseline."""

    # Load dataset
    if args.dataset == "swebench_lite":
        dataset_obj = SwebenchDataset(
            dataset="princeton-nlp/SWE-bench_Lite",
            split=args.split,
            filter_instance=args.filter_instance,
            repo_root=args.repo_cache_dir,
        )
    elif args.dataset == "locbench_v1":
        dataset_obj = LocbenchDataset(
            dataset="czlll/Loc-Bench_V1",
            split=args.split,
            filter_instance=args.filter_instance,
            repo_root=args.repo_cache_dir,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    dataset_instances = dataset_obj.load()
    if not dataset_instances:
        raise ValueError(f"No instances found in {args.dataset}")

    logger.info("Loaded %d instance(s)", len(dataset_instances))

    eval_path = args.eval_instances or str(
        Path.home() / ".codeminer" / f"swebench_lite_{args.split}_gt.json"
    )
    eval_metadata = dataset_obj.load_eval_metadata(eval_path)
    metrics_k = sorted(set(args.metrics_k))
    metric_max_k = max(metrics_k)
    aggregate = {}
    eval_count = 0
    all_results = [] if args.result_path else None

    for instance in dataset_instances:
        instance_id = instance["instance_id"]
        metadata = eval_metadata.get(instance_id)
        if not metadata:
            logger.info("Skipping %s - no eval metadata", instance_id)
            continue
        target_files, target_symbols = collect_targets(metadata)
        if not target_symbols:
            logger.info("Skipping %s - no valid target symbols", instance_id)
            continue

        vector_store = None
        try:
            t0 = time.time()

            # 1. Process instance (clone/checkout repo)
            dataset_obj.process_instance(instance)
            repo_path = dataset_obj.get_repo_path(instance)

            instance_dir_name = instance_id.replace("/", "__")
            index_path = str(Path(args.index_cache_dir) / instance_dir_name)

            # 2. Build or load vector store
            vector_store = build_hierarchical_vector_store(
                repo_path=repo_path,
                index_path=index_path,
                plan_name=None,
                languages=["python"],
                max_lines_per_chunk=300,
                build_levels=["l2"],
                embedding_model=args.embedding_model,
                embedding_provider=args.embedding_provider,
                embedding_dimension=args.embedding_dimension,
                embedding_kwargs={
                    "model_kwargs": {"trust_remote_code": True},
                },
                index_metric="ip",
            )

            # 3. Search
            query = instance["problem_statement"]
            search_results = vector_store.search_with_content(
                query, top_k=args.topk
            )

            # Convert to QueriedNode
            results = [
                QueriedNode(
                    node_name=n.node_name,
                    type=n.type,
                    file=n.file,
                    node_id=n.node_id,
                    start_line=n.start_line,
                    end_line=n.end_line,
                    score=n.score,
                    content=n.content,
                )
                for n in search_results
            ]

            elapsed = time.time() - t0

            # 4. Evaluate
            metrics = evaluate_predictions(
                nodes=results,
                target_files=target_files,
                target_symbols=target_symbols,
                ks=metrics_k,
            )
            aggregate_metrics(aggregate, metrics)
            eval_count += 1

            logger.info(
                "[%s] Done in %.1fs (%d results)", instance_id, elapsed, len(results),
            )
            for scope, per_k in metrics.items():
                for k, stats in per_k.items():
                    logger.info(
                        "  [%s] k=%d acc=%.3f prec=%.3f recall=%.3f hits=%d",
                        scope, k,
                        stats["accuracy"], stats["precision"],
                        stats["recall"], int(stats["hits"]),
                    )

            if all_results is not None:
                unique_files, normalized_symbols = extract_predictions(results)
                all_results.append({
                    "instance_id": instance_id,
                    "method": "embedding_baseline",
                    "topk": args.topk,
                    "num_results": len(results),
                    "elapsed_s": elapsed,
                    "metric_k_files": unique_files[:metric_max_k],
                    "metric_k_node_ids": normalized_symbols[:metric_max_k],
                    "metrics": metrics,
                })

        except Exception:
            logger.exception("Error processing %s", instance_id)
            continue
        finally:
            if vector_store is not None:
                vector_store.close()

    # ---- Aggregate ----
    if aggregate and eval_count:
        averaged = average_metrics(aggregate, eval_count)
        logger.info(
            "=== Embedding Baseline Aggregate (%d instances) ===", eval_count,
        )
        for scope, per_k in averaged.items():
            for k, stats in per_k.items():
                logger.info(
                    "[%s] k=%d acc=%.3f prec=%.3f recall=%.3f avg_hits=%.3f",
                    scope, k,
                    stats["accuracy"], stats["precision"],
                    stats["recall"], stats["avg_hits"],
                )

    if args.result_path and all_results is not None:
        result_path = Path(args.result_path).expanduser().resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", result_path)


def main():
    args = parse_args()
    logger.info("Dataset: %s", args.dataset)
    logger.info("Pipeline: Embedding(top%d)", args.topk)
    run_embedding_pipeline(args)


if __name__ == "__main__":
    main()
