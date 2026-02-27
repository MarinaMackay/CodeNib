#!/usr/bin/env python3
"""
Graph-based retrieval baseline script.

This script demonstrates a three-stage retrieval pipeline:
1. Stage 1: Select a small number of initial nodes (e.g., 5) using BM25.
2. Stage 2: Expand these nodes using the code graph (k-hop BFS) to gather
   a larger set of related nodes (e.g., up to 50).
3. Stage 3 (optional): Use embeddings within the expanded set to rank/select
   the final results.

This is a baseline for comparison with embedding-only retrieval.

Usage:
    # Graph-only (no embedding rerank)
    python examples/graph_retrieve_baseline.py --dataset swebench_lite

    # With embedding rerank on the expanded set
    python examples/graph_retrieve_baseline.py --dataset swebench_lite --embedding

    # Single instance
    python examples/graph_retrieve_baseline.py --dataset swebench_lite \\
        --filter-instance "^(astropy__astropy-12907)$"

    # Adjust stage parameters
    python examples/graph_retrieve_baseline.py --dataset swebench_lite \\
        --stage1-topk 5 --stage2-topk 50 --k-hop 2
"""
import argparse
import json
import time
from pathlib import Path

from codeminer.code_chunker import CodeChunker, RepoChunkingConfig
from codeminer.dataset.swebench import SwebenchDataset
from codeminer.dataset.locbench import LocbenchDataset
from codeminer.eval.retrieval_eval import (
    aggregate_metrics,
    average_metrics,
    collect_targets,
    evaluate_predictions,
    extract_predictions,
)
from codeminer.graph.roi_subgraph import ROISubgraph
from codeminer.index.embedding import CodeVectorStore, build_hierarchical_vector_store
from codeminer.index.sparse_idx.bm25_index import BM25CodeIndexer
from codeminer.log_utils import get_logger
from codeminer.scip_interface import SCIPIndexer
from codeminer.types import QueriedNode

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Graph-based retrieval baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["swebench_lite", "locbench_v1"],
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--filter-instance", type=str, default=".*")

    # Stage 1: BM25 seed selection
    parser.add_argument(
        "--stage1-topk", type=int, default=5,
        help="Number of seed nodes from BM25.",
    )
    # Stage 2: Graph expansion
    parser.add_argument(
        "--stage2-topk", type=int, default=50,
        help="Max nodes after graph expansion.",
    )
    parser.add_argument(
        "--k-hop", type=int, default=2,
        help="Number of hops for graph BFS expansion.",
    )
    # Stage 3: Optional embedding rerank
    parser.add_argument(
        "--embedding", action="store_true",
        help="Use embedding rerank within expanded set.",
    )
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


def run_graph_pipeline(args):
    """Run the graph-based retrieval baseline."""

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

        try:
            t0 = time.time()

            # 1. Process instance (clone/checkout repo)
            dataset_obj.process_instance(instance)
            repo_path = dataset_obj.get_repo_path(instance)

            instance_dir_name = instance_id.replace("/", "__")
            output_path = str(
                Path(args.index_cache_dir) / instance_dir_name
            )

            # 2. Build CodeGraph via SCIP indexing
            repo_indexer = SCIPIndexer(repo_path, output_dir=output_path)
            code_graph = repo_indexer.run_pipeline(
                project_name=instance_dir_name, skip_level="graph"
            )
            if code_graph is None:
                logger.error("Failed to build graph for %s", instance_id)
                continue

            # 3. Build BM25 index from graph
            bm25_index = BM25CodeIndexer(max_k=args.stage1_topk, language="english")
            bm25_index.build_index_from_graph(code_graph)

            # ---- Stage 1: BM25 seed selection ----
            query = instance["problem_statement"]
            bm25_results = bm25_index.search(query, top_k=args.stage1_topk)
            seed_names = [r.node_name for r in bm25_results]
            logger.info(
                "[%s] Stage 1: %d seed nodes from BM25",
                instance_id, len(seed_names),
            )

            # ---- Stage 2: Graph expansion (k-hop BFS) ----
            roi = ROISubgraph(code_graph)
            subgraph = roi.extract_subgraph(
                seed_names, k_hop=args.k_hop, direction="both"
            )
            expanded_nodes = roi.get_filtered_subgraph_nodes(
                subgraph, exclude_nodes=None, filter_tests=True
            )
            # Limit to stage2_topk
            expanded_nodes = expanded_nodes[: args.stage2_topk]
            logger.info(
                "[%s] Stage 2: %d nodes after %d-hop expansion",
                instance_id, len(expanded_nodes), args.k_hop,
            )

            # ---- Stage 3 (optional): Embedding rerank ----
            if args.embedding and expanded_nodes:
                index_path = str(Path(args.index_cache_dir) / instance_dir_name)
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
                # Build mask from expanded node IDs
                mask_ids = set()
                for node in expanded_nodes:
                    if node.node_name:
                        mask_ids.add(node.node_name)
                    if node.node_id:
                        mask_ids.add(node.node_id)

                reranked = vector_store.search(
                    query, top_k=args.stage2_topk, mask_node_ids=mask_ids
                )
                # Convert to QueriedNode
                results = [
                    QueriedNode(
                        node_name=n.node_name,
                        type=n.type,
                        file=n.file,
                        node_id=n.node_id or n.node_name,
                        start_line=n.start_line,
                        end_line=n.end_line,
                        score=n.score,
                        content=n.content,
                    )
                    for n in reranked
                ]
                vector_store.close()
                logger.info(
                    "[%s] Stage 3: %d nodes after embedding rerank",
                    instance_id, len(results),
                )
            else:
                # Convert expanded_nodes (NodeInfo) to QueriedNode.
                # ROISubgraph returns node_id=None; use node_name as
                # node_id since it has the "file:symbol()" format that
                # extract_predictions expects.
                results = [
                    QueriedNode(
                        node_name=n.node_name,
                        type=n.type,
                        file=n.file,
                        node_id=n.node_id or n.node_name,
                        start_line=n.start_line,
                        end_line=n.end_line,
                        score=n.score,
                        content=n.content,
                    )
                    for n in expanded_nodes
                ]

            elapsed = time.time() - t0

            # ---- Evaluate ----
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
                    "method": "graph_baseline",
                    "stage1_topk": args.stage1_topk,
                    "stage2_topk": args.stage2_topk,
                    "k_hop": args.k_hop,
                    "embedding_rerank": args.embedding,
                    "num_results": len(results),
                    "elapsed_s": elapsed,
                    "metric_k_files": unique_files[:metric_max_k],
                    "metric_k_node_ids": normalized_symbols[:metric_max_k],
                    "metrics": metrics,
                })

        except Exception:
            logger.exception("Error processing %s", instance_id)
            continue

    # ---- Aggregate ----
    if aggregate and eval_count:
        averaged = average_metrics(aggregate, eval_count)
        logger.info(
            "=== Graph Baseline Aggregate (%d instances) ===", eval_count,
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
    logger.info(
        "Pipeline: BM25(top%d) -> Graph(%d-hop, max %d) %s",
        args.stage1_topk, args.k_hop, args.stage2_topk,
        "-> Embedding rerank" if args.embedding else "",
    )
    run_graph_pipeline(args)


if __name__ == "__main__":
    main()
