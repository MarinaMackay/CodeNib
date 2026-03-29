#!/usr/bin/env python3
"""
Unified retrieval evaluation adapter for synthesized behavioral queries.

Evaluates synthesized queries (with ``gt_files`` / ``gt_symbols`` ground truth)
against any of 3 retrieval backends: BM25, Embedding, or Agent.
Reports Hit@K metrics per query and aggregated across the dataset.

Usage:
    # BM25
    python3 examples/eval_synthesized_queries.py \
        --pipeline bm25 \
        --queries-file filtered_behavioral_queries.json \
        --topk 50 --repo-cache-dir ~/.codeminer

    # Embedding
    python3 examples/eval_synthesized_queries.py \
        --pipeline embedding \
        --queries-file filtered_behavioral_queries.json \
        --embedding-model nomic-ai/CodeRankEmbed

    # Agent
    python3 examples/eval_synthesized_queries.py \
        --pipeline agent \
        --queries-file filtered_behavioral_queries.json \
        --model vertex_ai/gemini-2.5-flash --agent-mode hybrid

    # Single instance
    python3 examples/eval_synthesized_queries.py \
        --pipeline bm25 \
        --queries-file filtered_behavioral_queries.json \
        --filter-instance "astropy__astropy-6938"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the project root is on sys.path when running as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from codeminer.dataset.swebench import SwebenchDataset
from codeminer.eval.retrieval_eval import (
    aggregate_metrics,
    average_metrics,
    evaluate_predictions,
    extract_predictions,
    normalize_file_path,
    normalize_symbol_identifier,
)
from codeminer.log_utils import get_logger
from codeminer.types import QueriedNode

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate synthesized queries against retrieval backends.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=["bm25", "embedding", "agent"],
        help="Retrieval backend to evaluate.",
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        required=True,
        help="Path to synthesized queries JSON file.",
    )
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument(
        "--filter-instance",
        type=str,
        default=".*",
        help="Regex to filter instance_id values.",
    )
    parser.add_argument(
        "--metrics-k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 15, 20],
    )

    # Cache directories
    parser.add_argument(
        "--index-cache-dir",
        type=str,
        default="/mnt/data/codeminer",
    )
    parser.add_argument(
        "--repo-cache-dir",
        type=str,
        default="~/.codeminer/",
    )
    parser.add_argument("--result-path", type=str, default=None)

    # Embedding-specific
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nomic-ai/CodeRankEmbed",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default="huggingface",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=768,
    )

    # Agent-specific
    parser.add_argument(
        "--model",
        type=str,
        default="vertex_ai/gemini-2.5-flash",
        help="LiteLLM model name for agent pipeline.",
    )
    parser.add_argument(
        "--agent-mode",
        type=str,
        default="sparse",
        choices=["sparse", "dense", "hybrid"],
        help="Agent retrieval mode (determines which indexes to build).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Maximum agent turns per query.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Ground truth normalization
# ---------------------------------------------------------------------------


def collect_synthesized_targets(
    entry: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """Normalize ``gt_files`` and ``gt_symbols`` from a synthesized query entry.

    Unlike ``collect_targets()`` (which expects ``symbols_modified`` /
    ``symbols_deleted`` from SWE-bench metadata), this helper works
    directly with the synthesized query format.
    """
    raw_files: List[str] = entry.get("gt_files") or []
    raw_symbols: List[str] = entry.get("gt_symbols") or []

    normalized_files = [p for p in (normalize_file_path(f) for f in raw_files) if p]
    normalized_symbols = [
        s for s in (normalize_symbol_identifier(s) for s in raw_symbols) if s
    ]
    return normalized_files, normalized_symbols


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def build_bm25_pipeline(repo_path: str, index_path: str, topk: int) -> Any:
    """Build a BM25RetrievePipeline for the given repo."""
    from codeminer.model import BM25RetrievePipeline

    return BM25RetrievePipeline(
        repo_path=repo_path,
        index_path=index_path,
        top_k=topk,
        project_name=Path(index_path).name,
    )


def build_embedding_pipeline(
    repo_path: str,
    index_path: str,
    topk: int,
    embedding_model: str,
    embedding_provider: str,
    embedding_dimension: int,
) -> Any:
    """Build an EmbeddingRetrievePipeline for the given repo."""
    from codeminer.model import EmbeddingRetrievePipeline

    return EmbeddingRetrievePipeline(
        repo_path=repo_path,
        index_path=index_path,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        embedding_dimension=embedding_dimension,
        top_k=topk,
    )


def build_agent_runner(
    repo_path: str,
    cache_dir: str,
    args: argparse.Namespace,
) -> Any:
    """Build an AgentRunner for the given repo (Phase 1 + Phase 2 setup).

    Returns the AgentRunner instance ready for ``runner.run(query)``.
    """
    from codeminer.agent.runner import AgentRunner
    from codeminer.agent.skills.loader import SkillLoader
    from codeminer.agent.skills.registry import SkillRegistry
    from codeminer.compiler.index_builders import (
        BM25IndexBuilder,
        IndexBuilderRegistry,
        VectorIndexBuilder,
    )
    from codeminer.compiler.index_compiler import IndexCompiler, IndexCompilerConfig
    from codeminer.compiler.manifest import RepoManifest
    from codeminer.compiler.params import SessionContext
    from codeminer.index.embedding import CodeVectorStore
    from codeminer.index.sparse_idx.bm25_index import BM25CodeIndexer
    from codeminer.llm.litellm_chat import LiteLLMChat
    from codeminer.ops.rerank import RerankContext
    from codeminer.ops.retrieve import RetrieveContext

    # Phase 1: Index Compilation
    index_types = []
    if args.agent_mode in ("sparse", "hybrid"):
        index_types.append("bm25")
    if args.agent_mode in ("dense", "hybrid"):
        index_types.append("vector")

    builder_registry = IndexBuilderRegistry()
    builder_registry.register(
        "bm25",
        BM25IndexBuilder(languages=["python"]),
    )
    builder_registry.register(
        "vector",
        VectorIndexBuilder(
            languages=["python"],
            embedding_model=args.embedding_model,
            embedding_dimension=args.embedding_dimension,
        ),
    )

    config = IndexCompilerConfig(
        index_types=index_types,
        languages=["python"],
    )
    compiler = IndexCompiler(builder_registry, config)
    manifest = compiler.compile_repo(repo_path, cache_dir=cache_dir)

    # Phase 2: Load indexes and set up AgentRunner
    manifest_path = os.path.join(cache_dir, "repo_manifest.json")
    manifest = RepoManifest.load(manifest_path)

    bm25_index = None
    vector_store = None

    if "bm25" in manifest.indexes and manifest.indexes["bm25"].status == "fresh":
        bm25_index = BM25CodeIndexer()
        bm25_index.load_index(manifest.indexes["bm25"].path)

    if "vector" in manifest.indexes and manifest.indexes["vector"].status == "fresh":
        emb_model = manifest.indexes["vector"].config.get(
            "embedding_model",
            args.embedding_model,
        )
        emb_dim = manifest.indexes["vector"].config.get(
            "embedding_dimension",
            args.embedding_dimension,
        )
        vector_store = CodeVectorStore(
            embedding_model=emb_model,
            embedding_provider="huggingface",
            dimension=emb_dim,
            store_path=manifest.indexes["vector"].path,
        )
        vector_store.load(manifest.indexes["vector"].path)

    retrieve_ctx = RetrieveContext(
        bm25=bm25_index,
        vector_store=vector_store,
        default_top_k=args.topk,
        default_level="l2",
    )
    contexts: Dict[str, Any] = {"retrieve": retrieve_ctx}
    if vector_store:
        contexts["rerank"] = RerankContext(embedding_store=vector_store)

    skills_dir = os.path.join(_PROJECT_ROOT, "codeminer", "agent", "skills")
    loader = SkillLoader()
    loaded = loader.load_all(skills_dir, contexts=contexts)
    logger.info("Loaded %d skills: %s", len(loaded), [s.skill_id for s in loaded])

    session_ctx = SessionContext(
        repo_path=repo_path,
        repo_size=manifest.file_count,
        primary_language=manifest.languages[0] if manifest.languages else "python",
    )

    llm = LiteLLMChat(
        model=args.model,
        temperature=0.0,
        max_tokens=1024,
    )

    runner = AgentRunner(
        llm=llm,
        registry=SkillRegistry(),
        max_turns=args.max_turns,
        manifest=manifest,
        session_ctx=session_ctx,
    )
    return runner


def run_agent_query(runner: Any, query: str) -> List[QueriedNode]:
    """Run an agent query and extract QueriedNode results from tool calls."""
    result = runner.run(query)
    all_nodes: List[QueriedNode] = []
    for tc in result.tool_calls:
        if isinstance(tc.result, list):
            for item in tc.result:
                if isinstance(item, QueriedNode):
                    all_nodes.append(item)
    return all_nodes


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_eval(args: argparse.Namespace) -> None:
    """Main evaluation loop."""

    # Load queries
    queries_path = Path(args.queries_file).expanduser().resolve()
    with open(queries_path, "r", encoding="utf-8") as f:
        all_queries: List[Dict[str, Any]] = json.load(f)
    logger.info("Loaded %d queries from %s", len(all_queries), queries_path)

    # Filter by instance regex
    instance_pattern = re.compile(args.filter_instance)
    all_queries = [
        q for q in all_queries if instance_pattern.search(q.get("instance_id", ""))
    ]
    logger.info("After filtering: %d queries", len(all_queries))

    if not all_queries:
        logger.warning("No queries matched filter '%s'", args.filter_instance)
        return

    # Group queries by instance_id (to cache pipeline per instance)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for q in all_queries:
        groups[q["instance_id"]].append(q)
    logger.info("Grouped into %d instance(s)", len(groups))

    # Dataset helper for checkout
    dataset_obj = SwebenchDataset(
        dataset="princeton-nlp/SWE-bench_Lite",
        split="test",
        filter_instance=".*",
        repo_root=args.repo_cache_dir,
    )

    metrics_k = sorted(set(args.metrics_k))
    metric_max_k = max(metrics_k)
    aggregate: Dict[str, Dict[int, Dict[str, float]]] = {}
    eval_count = 0
    all_results: Optional[List[Dict[str, Any]]] = [] if args.result_path else None

    for instance_id, queries in groups.items():
        # All queries in a group share repo/base_commit — use the first
        representative = queries[0]
        target_instance = {
            "repo": representative["repo"],
            "instance_id": representative["instance_id"],
            "base_commit": representative["base_commit"],
        }

        pipeline = None
        try:
            # Checkout repo at the correct commit
            dataset_obj.process_instance(target_instance)
            repo_path = dataset_obj.get_repo_path(target_instance)
            index_path = str(
                Path(args.index_cache_dir) / instance_id.replace("/", "__")
            )

            # Build pipeline once per instance group
            t0_pipeline = time.time()
            if args.pipeline == "bm25":
                pipeline = build_bm25_pipeline(repo_path, index_path, args.topk)
            elif args.pipeline == "embedding":
                pipeline = build_embedding_pipeline(
                    repo_path,
                    index_path,
                    args.topk,
                    args.embedding_model,
                    args.embedding_provider,
                    args.embedding_dimension,
                )
            elif args.pipeline == "agent":
                cache_dir = os.path.join(index_path, ".codeminer_cache")
                pipeline = build_agent_runner(repo_path, cache_dir, args)

            pipeline_elapsed = time.time() - t0_pipeline
            logger.info(
                "[%s] Pipeline built in %.1fs",
                instance_id,
                pipeline_elapsed,
            )

            # Evaluate each query in this instance group
            for entry in queries:
                query_id = entry.get("query_id", instance_id)
                query_text = entry["query"]
                target_files, target_symbols = collect_synthesized_targets(entry)

                if not target_files and not target_symbols:
                    logger.info(
                        "Skipping %s - no ground truth targets",
                        query_id,
                    )
                    continue

                t0 = time.time()
                if args.pipeline == "agent":
                    results = run_agent_query(pipeline, query_text)
                else:
                    results = pipeline.query(query_text)
                elapsed = time.time() - t0

                metrics = evaluate_predictions(
                    nodes=results,
                    target_files=target_files,
                    target_symbols=target_symbols,
                    ks=metrics_k,
                )
                aggregate_metrics(aggregate, metrics)
                eval_count += 1

                logger.info(
                    "[%s] Done in %.1fs (%d results)",
                    query_id,
                    elapsed,
                    len(results),
                )
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

                if all_results is not None:
                    unique_files, normalized_symbols = extract_predictions(results)
                    all_results.append(
                        {
                            "query_id": query_id,
                            "instance_id": instance_id,
                            "method": f"{args.pipeline}_synthesized",
                            "topk": args.topk,
                            "num_results": len(results),
                            "elapsed_s": elapsed,
                            "target_files": target_files,
                            "target_symbols": target_symbols,
                            "metric_k_files": unique_files[:metric_max_k],
                            "metric_k_node_ids": normalized_symbols[:metric_max_k],
                            "metrics": metrics,
                        }
                    )

        except Exception:
            logger.exception("Error processing instance %s", instance_id)
            continue
        finally:
            if pipeline is not None and hasattr(pipeline, "close"):
                pipeline.close()

    # ---- Aggregate ----
    if aggregate and eval_count:
        averaged = average_metrics(aggregate, eval_count)
        logger.info(
            "=== %s Synthesized Query Aggregate (%d queries) ===",
            args.pipeline.upper(),
            eval_count,
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
    else:
        logger.warning("No queries were evaluated.")

    if args.result_path and all_results is not None:
        result_path = Path(args.result_path).expanduser().resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", result_path)


def main() -> None:
    args = parse_args()
    logger.info("Pipeline: %s", args.pipeline)
    logger.info("Queries file: %s", args.queries_file)
    logger.info("Top-K: %d", args.topk)
    run_eval(args)


if __name__ == "__main__":
    main()
