#!/usr/bin/env python3
"""
Evaluate synthesized behavioral queries against retrieval backends.

Thin adapter that reuses the existing pipeline classes (BM25RetrievePipeline,
EmbeddingRetrievePipeline, AgentRunner) and evaluation utilities without
duplicating any logic.  The only new code is the glue that loads the
synthesized JSON format and feeds it through the standard eval functions.

Usage:
    # BM25
    python examples/eval_synthesized_queries.py \
        --pipeline bm25 \
        --queries-file filtered_behavioral_queries.json

    # Embedding
    python examples/eval_synthesized_queries.py \
        --pipeline embedding \
        --queries-file filtered_behavioral_queries.json

    # Agent
    python examples/eval_synthesized_queries.py \
        --pipeline agent \
        --queries-file filtered_behavioral_queries.json \
        --model vertex_ai/gemini-2.5-flash --agent-mode hybrid

    # Single instance
    python examples/eval_synthesized_queries.py \
        --pipeline bm25 \
        --queries-file filtered_behavioral_queries.json \
        --filter-instance "astropy__astropy-6938"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate synthesized queries against retrieval backends.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pipeline", required=True, choices=["bm25", "embedding", "agent"])
    p.add_argument("--queries-file", required=True, help="Synthesized queries JSON.")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--filter-instance", type=str, default=".*")
    p.add_argument("--metrics-k", type=int, nargs="+", default=[1, 3, 5, 10, 15, 20])
    p.add_argument("--index-cache-dir", type=str, default="/mnt/data/codeminer")
    p.add_argument("--repo-cache-dir", type=str, default="~/.codeminer/")
    p.add_argument("--result-path", type=str, default=None)

    # Embedding
    p.add_argument("--embedding-model", default="nomic-ai/CodeRankEmbed")
    p.add_argument("--embedding-provider", default="huggingface")
    p.add_argument("--embedding-dimension", type=int, default=768)

    # Agent
    p.add_argument("--model", default="vertex_ai/gemini-2.5-flash")
    p.add_argument(
        "--agent-mode", default="sparse", choices=["sparse", "dense", "hybrid"]
    )
    p.add_argument("--max-turns", type=int, default=5)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Ground-truth adapter
# ---------------------------------------------------------------------------


def collect_synthesized_targets(
    entry: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """Normalize gt_files / gt_symbols from the synthesized query format."""
    raw_files: list = entry.get("gt_files") or []
    raw_symbols: list = entry.get("gt_symbols") or []
    files = [p for p in (normalize_file_path(f) for f in raw_files) if p]
    symbols = [s for s in (normalize_symbol_identifier(s) for s in raw_symbols) if s]
    return files, symbols


# ---------------------------------------------------------------------------
# Pipeline factories — each returns (pipeline_or_runner, query_fn, close_fn)
# ---------------------------------------------------------------------------


def _make_bm25(repo_path: str, index_path: str, args: argparse.Namespace):
    from codeminer.model.bm25_retrieve_pipeline import BM25RetrievePipeline

    pipe = BM25RetrievePipeline(
        repo_path=repo_path,
        index_path=index_path,
        top_k=args.topk,
        project_name=Path(index_path).name,
    )
    return pipe, pipe.query, pipe.close


def _make_embedding(repo_path: str, index_path: str, args: argparse.Namespace):
    from codeminer.model.embedding_retrieve_pipeline import EmbeddingRetrievePipeline

    pipe = EmbeddingRetrievePipeline(
        repo_path=repo_path,
        index_path=index_path,
        embedding_model=args.embedding_model,
        embedding_provider=args.embedding_provider,
        embedding_dimension=args.embedding_dimension,
        top_k=args.topk,
    )
    return pipe, pipe.query, pipe.close


def _make_agent(repo_path: str, index_path: str, args: argparse.Namespace):
    import os

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

    cache_dir = os.path.join(index_path, ".codeminer_cache")

    # Phase 1: compile indexes
    idx_types = []
    if args.agent_mode in ("sparse", "hybrid"):
        idx_types.append("bm25")
    if args.agent_mode in ("dense", "hybrid"):
        idx_types.append("vector")

    breg = IndexBuilderRegistry()
    breg.register("bm25", BM25IndexBuilder(languages=["python"]))
    breg.register(
        "vector",
        VectorIndexBuilder(
            languages=["python"],
            embedding_model=args.embedding_model,
            embedding_dimension=args.embedding_dimension,
        ),
    )
    compiler = IndexCompiler(
        breg,
        IndexCompilerConfig(
            index_types=idx_types,
            languages=["python"],
        ),
    )
    compiler.compile_repo(repo_path, cache_dir=cache_dir)

    # Phase 2: load indexes, skills, build runner
    manifest = RepoManifest.load(os.path.join(cache_dir, "repo_manifest.json"))
    bm25_index, vector_store = None, None

    if "bm25" in manifest.indexes and manifest.indexes["bm25"].status == "fresh":
        bm25_index = BM25CodeIndexer()
        bm25_index.load_index(manifest.indexes["bm25"].path)

    if "vector" in manifest.indexes and manifest.indexes["vector"].status == "fresh":
        cfg = manifest.indexes["vector"].config
        vector_store = CodeVectorStore(
            embedding_model=cfg.get("embedding_model", args.embedding_model),
            embedding_provider="huggingface",
            dimension=cfg.get("embedding_dimension", args.embedding_dimension),
            store_path=manifest.indexes["vector"].path,
        )
        vector_store.load(manifest.indexes["vector"].path)

    ctx: Dict[str, Any] = {
        "retrieve": RetrieveContext(
            bm25=bm25_index,
            vector_store=vector_store,
            default_top_k=args.topk,
            default_level="l2",
        ),
    }
    if vector_store:
        ctx["rerank"] = RerankContext(embedding_store=vector_store)

    skills_dir = os.path.join(_PROJECT_ROOT, "codeminer", "agent", "skills")
    SkillLoader().load_all(skills_dir, contexts=ctx)

    runner = AgentRunner(
        llm=LiteLLMChat(model=args.model, temperature=0.0, max_tokens=1024),
        registry=SkillRegistry(),
        max_turns=args.max_turns,
        manifest=manifest,
        session_ctx=SessionContext(
            repo_path=repo_path,
            repo_size=manifest.file_count,
            primary_language=manifest.languages[0] if manifest.languages else "python",
        ),
    )

    def query_fn(q: str, **_kw) -> List[QueriedNode]:
        result = runner.run(q)
        nodes: List[QueriedNode] = []
        for tc in result.tool_calls:
            if isinstance(tc.result, list):
                nodes.extend(n for n in tc.result if isinstance(n, QueriedNode))
        return nodes

    return runner, query_fn, lambda: None


_PIPELINE_FACTORIES = {
    "bm25": _make_bm25,
    "embedding": _make_embedding,
    "agent": _make_agent,
}


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------


def run_eval(args: argparse.Namespace) -> None:
    queries_path = Path(args.queries_file).expanduser().resolve()
    with open(queries_path, encoding="utf-8") as f:
        all_queries: List[Dict[str, Any]] = json.load(f)
    logger.info("Loaded %d queries from %s", len(all_queries), queries_path)

    pat = re.compile(args.filter_instance)
    all_queries = [q for q in all_queries if pat.search(q.get("instance_id", ""))]
    logger.info("After filtering: %d queries", len(all_queries))
    if not all_queries:
        return

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for q in all_queries:
        groups[q["instance_id"]].append(q)
    logger.info("Grouped into %d instance(s)", len(groups))

    dataset_obj = SwebenchDataset(
        dataset="princeton-nlp/SWE-bench_Lite",
        split="test",
        filter_instance=".*",
        repo_root=args.repo_cache_dir,
    )
    factory = _PIPELINE_FACTORIES[args.pipeline]
    metrics_k = sorted(set(args.metrics_k))
    metric_max_k = max(metrics_k)
    aggregate: Dict = {}
    eval_count = 0
    all_results: Optional[List[Dict]] = [] if args.result_path else None

    for instance_id, queries in groups.items():
        rep = queries[0]
        inst = {
            "repo": rep["repo"],
            "instance_id": rep["instance_id"],
            "base_commit": rep["base_commit"],
        }

        pipe, query_fn, close_fn = None, None, None
        try:
            dataset_obj.process_instance(inst)
            repo_path = dataset_obj.get_repo_path(inst)
            index_path = str(
                Path(args.index_cache_dir) / instance_id.replace("/", "__")
            )

            t0 = time.time()
            pipe, query_fn, close_fn = factory(repo_path, index_path, args)
            logger.info("[%s] Pipeline built in %.1fs", instance_id, time.time() - t0)

            for entry in queries:
                qid = entry.get("query_id", instance_id)
                tgt_files, tgt_syms = collect_synthesized_targets(entry)
                if not tgt_files and not tgt_syms:
                    logger.info("Skipping %s — no ground truth", qid)
                    continue

                t0 = time.time()
                results = query_fn(entry["query"])
                elapsed = time.time() - t0

                metrics = evaluate_predictions(results, tgt_files, tgt_syms, metrics_k)
                aggregate_metrics(aggregate, metrics)
                eval_count += 1

                logger.info("[%s] %.1fs  %d results", qid, elapsed, len(results))
                for scope, per_k in metrics.items():
                    for k, st in per_k.items():
                        logger.info(
                            "  [%s] k=%d acc=%.3f prec=%.3f rec=%.3f hits=%d",
                            scope,
                            k,
                            st["accuracy"],
                            st["precision"],
                            st["recall"],
                            int(st["hits"]),
                        )

                if all_results is not None:
                    uf, ns = extract_predictions(results)
                    all_results.append(
                        {
                            "query_id": qid,
                            "instance_id": instance_id,
                            "method": f"{args.pipeline}_synthesized",
                            "topk": args.topk,
                            "num_results": len(results),
                            "elapsed_s": elapsed,
                            "target_files": tgt_files,
                            "target_symbols": tgt_syms,
                            "metric_k_files": uf[:metric_max_k],
                            "metric_k_node_ids": ns[:metric_max_k],
                            "metrics": metrics,
                        }
                    )
        except Exception:
            logger.exception("Error processing %s", instance_id)
        finally:
            if close_fn:
                close_fn()

    if aggregate and eval_count:
        avg = average_metrics(aggregate, eval_count)
        logger.info(
            "=== %s Aggregate (%d queries) ===", args.pipeline.upper(), eval_count
        )
        for scope, per_k in avg.items():
            for k, st in per_k.items():
                logger.info(
                    "[%s] k=%d acc=%.3f prec=%.3f rec=%.3f avg_hits=%.3f",
                    scope,
                    k,
                    st["accuracy"],
                    st["precision"],
                    st["recall"],
                    st["avg_hits"],
                )

    if args.result_path and all_results is not None:
        out = Path(args.result_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", out)


if __name__ == "__main__":
    args = parse_args()
    logger.info("Pipeline: %s  TopK: %d", args.pipeline, args.topk)
    run_eval(args)
