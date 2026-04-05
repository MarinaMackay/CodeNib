#!/usr/bin/env python3
"""
Single-skill evaluation script for AgentRunner with BM25 search.

This script evaluates BM25 search skill on SWE-bench dataset using cloud LLMs
(Gemini 2.5 Flash, etc.) for tool-calling. It tests the Agent + Skill + Eval
pipeline end-to-end, independent of the LLM provider.

Usage:
    # Smoke (test split, optional external GT JSON; file is auto-built if missing)
    python examples/skill_agent_eval.py \
        --split test \
        --filter-instance "^(astropy__astropy-12907)$" \
        --eval-instances "$HOME/.codeminer/swebench_lite_test_gt_single.json" \
        --result-path "$HOME/skill_eval_test.json"

    # Vertex project/region (optional; else GOOGLE_CLOUD_PROJECT / litellm defaults)
    python examples/skill_agent_eval.py \
        --vertex-project YOUR_GCP_PROJECT \
        --vertex-location us-central1 \
        ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the project root is on sys.path when running as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from codeminer.agent.skills.loader import SkillLoader
from codeminer.agent.skills.registry import SkillRegistry
from codeminer.code_chunker import CodeChunker, RepoChunkingConfig
from codeminer.dataset.swebench import SwebenchDataset
from codeminer.eval.retrieval_eval import (
    aggregate_metrics,
    average_metrics,
    collect_targets,
    evaluate_predictions,
)
from codeminer.index.sparse_idx.bm25_index import BM25CodeIndexer
from codeminer.llm.litellm_chat import LiteLLMChat
from codeminer.log_utils import get_logger
from codeminer.ops.retrieve import RetrieveContext
from codeminer.types import QueriedNode

logger = get_logger(__name__)



# Output Schema 

@dataclass
class CodeSymbol:
    """A code symbol with its location (from hengjia branch)."""

    name: str  # Symbol name, e.g. "Foo::bar()"
    type: str  # function / class / method / field
    file_path: str  # Relative to repo root
    line_start: int
    line_end: int
    action: str  # modify / add / delete
    description: str = ""  # Brief description


@dataclass
class LocResult:
    """Localization result for a single instance (from hengjia branch)."""

    success: bool
    repo_path: str
    locations: List[CodeSymbol] = field(default_factory=list)
    error_message: Optional[str] = None
    execution_log: List[str] = field(default_factory=list)
    usage: Optional[Dict[str, Any]] = None


@dataclass
class SkillEvalReport:
    """Evaluation report for skill-based retrieval."""

    dataset: str
    model: str
    skill_ids: List[str]
    instance_count: int
    results: List[Dict[str, Any]] = field(default_factory=list)
    aggregate_metrics: Dict[str, Any] = field(default_factory=dict)


# QueriedNode -> CodeSymbol conversion

def queried_node_to_symbol(node: QueriedNode) -> CodeSymbol:
    """Convert a QueriedNode to CodeSymbol."""
    return CodeSymbol(
        name=node.node_name or "unknown",
        type=node.type or "unknown",
        file_path=node.file or "",
        line_start=node.start_line or 0,
        line_end=node.end_line or 0,
        action="modify",  # Default to modify, agent can't determine action
        description=f"score={node.score:.4f}" if node.score else "",
    )


# Core evaluation logic

def build_bm25_index(
    repo_path: str, languages: List[str], max_k: int = 128
) -> BM25CodeIndexer:
    """Chunk the repo and build a BM25 index."""
    primary = languages[0] if languages else "python"
    chunker = CodeChunker(
        language=primary,
        repo_config=RepoChunkingConfig(languages=languages),
        max_lines_per_chunk=300,
    )
    chunks = chunker.chunk_repository(repo_path=repo_path)
    if not chunks:
        raise RuntimeError(f"No code chunks generated from {repo_path}")
    logger.info(f"  BM25: indexed {len(chunks)} chunks")
    return BM25CodeIndexer(chunks=chunks, max_k=max_k)


def run_agent_with_bm25(
    query: str,
    bm25_index: BM25CodeIndexer,
    llm: LiteLLMChat,
    top_k: int = 10,
    max_turns: int = 3,
    repo_path: str = ".",
) -> tuple[List[QueriedNode], List[str], Optional[Dict[str, Any]]]:
    """
    Run AgentRunner with BM25 search skill.

    Uses the full AgentRunner pipeline with LLM tool-calling to let the
    model decide when and how to use the bm25_search skill.

    Returns:
        Tuple of (results, execution_log, usage_stats)
    """
    from codeminer.agent.runner import AgentRunner
    from codeminer.compiler.params import SessionContext

    execution_log = []
    usage_stats = {}

    try:
        # Load skills
        skills_dir = os.path.join(_PROJECT_ROOT, "codeminer", "agent", "skills")
        retrieve_ctx = RetrieveContext(
            bm25=bm25_index,
            default_top_k=top_k,
            default_level="l2",
        )
        loader = SkillLoader()
        loaded = loader.load_all(skills_dir, contexts={"retrieve": retrieve_ctx})
        execution_log.append(f"Loaded {len(loaded)} skills")

        # Create session context
        session_ctx = SessionContext(
            repo_path=repo_path,
            repo_size=1000,  # Placeholder
            primary_language="python",
        )

        # Create AgentRunner (only allow bm25_search skill)
        registry = SkillRegistry()
        all_skills = set(registry.list_skills())
        exclude_skills = all_skills - {"bm25_search"}

        runner = AgentRunner(
            llm=llm,
            registry=registry,
            max_turns=max_turns,
            exclude_skills=exclude_skills,
            session_ctx=session_ctx,
        )
        execution_log.append(f"Created AgentRunner (max_turns={max_turns})")

        # Run the agent
        result = runner.run(query)
        execution_log.append(f"Agent completed in {result.total_turns} turns")
        execution_log.append(f"Total tool calls: {len(result.tool_calls)}")

        # Collect results from tool calls
        all_results = []
        for tc in result.tool_calls:
            execution_log.append(f"  - {tc.skill_id}({tc.arguments})")
            if tc.error:
                execution_log.append(f"    Error: {tc.error}")
            elif isinstance(tc.result, list):
                all_results.extend(tc.result)
                execution_log.append(f"    Got {len(tc.result)} results")

        # Collect usage stats
        usage_stats = {
            "total_turns": result.total_turns,
            "total_duration_ms": result.total_duration_ms,
            "tool_call_count": len(result.tool_calls),
        }

        return all_results, execution_log, usage_stats

    except Exception as e:
        execution_log.append(f"Error: {str(e)}")
        logger.error(f"Failed to run agent: {e}", exc_info=True)
        return [], execution_log, usage_stats


def evaluate_instance(
    instance: Dict[str, Any],
    dataset: SwebenchDataset,
    llm: LiteLLMChat,
    args: argparse.Namespace,
    eval_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate a single SWE-bench instance."""
    instance_id = instance["instance_id"]
    problem_statement = instance["problem_statement"]

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Instance: {instance_id}")
    logger.info(f"{'=' * 60}")

    start_time = time.time()
    execution_log = []

    try:
        # Step 1: Process instance (clone repo, checkout commit)
        execution_log.append(f"Processing instance {instance_id}")
        dataset.process_instance(instance)
        repo_path = dataset.get_repo_path(instance)
        execution_log.append(f"Repository prepared at {repo_path}")

        # Step 2: Build BM25 index
        execution_log.append("Building BM25 index...")
        bm25_index = build_bm25_index(repo_path, args.languages, max_k=128)
        execution_log.append(f"BM25 index built with {len(bm25_index.documents)} documents")

        # Step 3: Run agent with BM25 search
        execution_log.append("Running agent with BM25 search...")
        results, search_log, usage = run_agent_with_bm25(
            query=problem_statement,
            bm25_index=bm25_index,
            llm=llm,
            top_k=max(args.metrics_k),  # Use max K for evaluation
            max_turns=args.max_turns,
            repo_path=repo_path,
        )
        execution_log.extend(search_log)

        # Step 4: Convert to CodeSymbol
        locations = [queried_node_to_symbol(node) for node in results]
        execution_log.append(f"Converted {len(locations)} nodes to CodeSymbol")

        # Step 5: Evaluate predictions (HF row and/or external GT JSON)
        if eval_metadata is not None:
            meta = eval_metadata.get(instance_id)
            if not meta:
                raise RuntimeError(
                    f"No eval metadata for {instance_id} in --eval-instances"
                )
            target_files, target_symbols = collect_targets(
                meta, simplified_symbols=True
            )
        else:
            target_files, target_symbols = collect_targets(
                instance, simplified_symbols=True
            )
        execution_log.append(
            f"Target: {len(target_files)} files, {len(target_symbols)} symbols"
        )

        gt_empty = not target_files and not target_symbols
        if gt_empty:
            logger.warning(
                "%s: GT has no target_files/target_symbols (HF rows often lack these). "
                "Retrieval metrics are not meaningful without --eval-instances (gt_locate JSON).",
                instance_id,
            )

        metrics = evaluate_predictions(
            nodes=results,
            target_files=target_files,
            target_symbols=target_symbols,
            ks=args.metrics_k,
        )

        loc_result = LocResult(
            success=True,
            repo_path=repo_path,
            locations=locations,
            execution_log=execution_log,
            usage=usage,
        )

        elapsed = time.time() - start_time
        logger.info(f"Completed in {elapsed:.2f}s")

        return {
            "instance_id": instance_id,
            "success": True,
            "loc_result": asdict(loc_result),
            "metrics": metrics,
            "target_files": target_files,
            "target_symbols": target_symbols,
            "metrics_meaningful": not gt_empty,
            "elapsed_seconds": elapsed,
        }

    except Exception as e:
        logger.error(f"Failed to evaluate {instance_id}: {e}", exc_info=True)
        elapsed = time.time() - start_time

        loc_result = LocResult(
            success=False,
            repo_path=dataset.get_repo_path(instance),
            error_message=str(e),
            execution_log=execution_log,
        )

        return {
            "instance_id": instance_id,
            "success": False,
            "loc_result": asdict(loc_result),
            "error": str(e),
            "elapsed_seconds": elapsed,
        }


# Main evaluation loop

def run_evaluation(args: argparse.Namespace) -> SkillEvalReport:
    """Run evaluation on the dataset."""
    logger.info(f"Starting evaluation with:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Split: {args.split}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Filter: {args.filter_instance}")
    logger.info(f"  Metrics K: {args.metrics_k}")
    if args.eval_instances:
        logger.info(f"  Eval instances: {args.eval_instances}")
    else:
        logger.warning(
            "No --eval-instances: default HF instances usually lack symbol-level GT; "
            "metrics may stay at zero. Use a gt_locate JSON path for valid retrieval_eval."
        )

    # Load dataset
    dataset = SwebenchDataset(
        dataset=args.dataset,
        split=args.split,
        filter_instance=args.filter_instance,
        root=args.cache_dir,
        repo_root=args.repo_cache_dir,
    )
    instances = dataset.load()
    logger.info(f"Loaded {len(instances)} instances")

    eval_lookup: Optional[Dict[str, Any]] = None
    if args.eval_instances:
        eval_lookup = dataset.load_eval_metadata(args.eval_instances)

    # Initialize LLM
    llm_kwargs = {}
    if args.api_base:
        llm_kwargs["api_base"] = args.api_base
    if args.api_key:
        llm_kwargs["api_key"] = args.api_key

    vertex_extra: Dict[str, Any] = {}
    if args.vertex_project:
        vertex_extra["vertex_project"] = args.vertex_project
    if args.vertex_location:
        vertex_extra["vertex_location"] = args.vertex_location

    llm = LiteLLMChat(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_kwargs=vertex_extra,
        **llm_kwargs,
    )

    # Run evaluation
    results = []
    aggregate = {}
    metric_count = 0

    for idx, instance in enumerate(instances):
        logger.info(f"\n[{idx + 1}/{len(instances)}] Evaluating {instance['instance_id']}")

        result = evaluate_instance(
            instance, dataset, llm, args, eval_metadata=eval_lookup
        )
        results.append(result)

        # Aggregate metrics
        if result["success"] and "metrics" in result:
            aggregate_metrics(aggregate, result["metrics"])
            metric_count += 1

    # Compute average metrics (only over instances that produced metrics)
    avg_metrics = (
        average_metrics(aggregate, metric_count) if metric_count else {}
    )

    # Build report
    report = SkillEvalReport(
        dataset=args.dataset,
        model=args.model,
        skill_ids=["bm25_search"],
        instance_count=len(instances),
        results=results,
        aggregate_metrics=avg_metrics,
    )

    return report


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-skill evaluation for BM25 search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset args
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="SWE-bench dataset to use",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="HuggingFace dataset split (e.g. test, dev)",
    )
    parser.add_argument(
        "--filter-instance",
        type=str,
        default=".*",
        help="Regex filter for instance_id",
    )
    parser.add_argument(
        "--eval-instances",
        type=str,
        default=None,
        help=(
            "Path to GT/eval JSON (instance_id -> targets). "
            "If missing, SwebenchDataset generates it (clone + patch parse; slow)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache dataset files",
    )
    parser.add_argument(
        "--repo-cache-dir",
        type=str,
        default=None,
        help="Directory to cache repositories",
    )

    # Model args
    parser.add_argument(
        "--model",
        type=str,
        default="vertex_ai/gemini-2.5-flash",
        help="LLM model to use (litellm format)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL (for vLLM, etc.)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (optional)",
    )
    parser.add_argument(
        "--vertex-project",
        type=str,
        default=None,
        help="Vertex AI GCP project id (passed to litellm; optional if env is set)",
    )
    parser.add_argument(
        "--vertex-location",
        type=str,
        default=None,
        help="Vertex AI region, e.g. us-central1 (optional)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens for LLM response",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=3,
        help="Max turns for agent execution",
    )

    # Evaluation args
    parser.add_argument(
        "--metrics-k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="K values for accuracy@K and recall@K",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["python"],
        help="Programming languages to index",
    )

    # Output args
    parser.add_argument(
        "--result-path",
        type=str,
        default="results/skill_eval.json",
        help="Path to save evaluation results",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Run evaluation
    report = run_evaluation(args)

    # Save results
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with open(result_path, "w") as f:
        json.dump(asdict(report), f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info("Evaluation complete!")
    logger.info(f"{'=' * 60}")
    logger.info(f"Results saved to: {result_path}")
    logger.info(f"\nAggregate Metrics:")

    # Print summary
    for scope in ["files", "symbols"]:
        if scope in report.aggregate_metrics:
            logger.info(f"\n{scope.upper()}:")
            for k, stats in sorted(report.aggregate_metrics[scope].items()):
                logger.info(f"  @{k}:")
                logger.info(f"    accuracy: {stats['accuracy']:.4f}")
                logger.info(f"    recall:   {stats['recall']:.4f}")
                logger.info(f"    precision:{stats['precision']:.4f}")


if __name__ == "__main__":
    main()
