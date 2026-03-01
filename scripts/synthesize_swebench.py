#!/usr/bin/env python3
"""
Synthesize natural-language queries from collected SWE-bench instances.

python3 scripts/synthesize_swebench.py \
  --dataset swebench_lite \
  --split test \
  --instance-id "astropy__astropy-6938" \
  --model-name opus \
  --query-types behavioral \
  --allowed-tools "Read,Grep,Glob,Bash" \
  --behavioral-consensus-runs 1 \
  --output-dir ./synthesis_output \
  --cache-dir ~/.codeminer \
  --repo-cache-dir ~/.codeminer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from codeminer.dataset.swebench import SwebenchDataset
from codeminer.dataset.synthesize import ClaudeQuerySynthesizer
from codeminer.dataset.utils import QueryType
from codeminer.log_utils import get_logger

logger = get_logger(__name__)


def _dump_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    return data


def _parse_query_types(values: List[str]) -> List[QueryType]:
    tokens: List[str] = []
    for value in values:
        tokens.extend(part.strip() for part in value.split(",") if part.strip())
    if not tokens:
        tokens = [QueryType.BEHAVIORAL.value]
    return [QueryType(token) for token in tokens]


def _to_compact_record(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "repo": item.get("repo"),
        "instance_id": item.get("instance_id"),
        "base_commit": item.get("base_commit"),
        "query": item.get("query"),
        "category": item.get("query_type") or item.get("difficulty"),
        "gt_symbols": item.get("target_symbols") or [],
        "gt_symbol_nodes": item.get("target_symbol_nodes") or [],
        "gt_files": item.get("target_files") or [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthesize natural-language queries for sampled SWE-bench instances."
    )
    parser.add_argument(
        "--selected-instances",
        type=str,
        default=None,
        help="Path to selected_instances.json",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="swebench_lite",
        choices=["swebench_lite", "swebench_verified"],
        help="Dataset to use when loading directly (default: swebench_lite)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--filter-instance",
        type=str,
        default=".*",
        help="Regex pattern to filter instances (default: .* for all)",
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--repo-cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--instance-id",
        type=str,
        default=None,
        help="Debug mode: synthesize only this instance_id.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="opus",
        help="Claude agent model name (e.g., sonnet, opus).",
    )
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic block sampling.",
    )
    parser.add_argument(
        "--behavioral-consensus-runs",
        type=int,
        default=3,
        help="Number of behavioral generation passes for GT consensus voting.",
    )
    parser.add_argument(
        "--permission-mode",
        type=str,
        default="bypassPermissions",
        help="Claude agent permission mode.",
    )
    parser.add_argument(
        "--allowed-tools",
        type=str,
        default="Read,Grep,Glob,Bash",
        help="Comma-separated list of allowed tools.",
    )
    parser.add_argument(
        "--synthesis-limit",
        type=int,
        default=None,
        help="Synthesize only first N instances.",
    )
    parser.add_argument(
        "--repeat-per-instance",
        type=int,
        default=1,
        help="Generate multiple outputs per instance for debugging.",
    )
    parser.add_argument(
        "--query-types",
        nargs="*",
        default=[QueryType.BEHAVIORAL.value],
        help=(
            "Query types to synthesize (space or comma separated): "
            "behavioral,module_hint,file_hint,symbol_hint,reasoning "
            "(default: behavioral)"
        ),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="synthesized_queries.json",
        help="Output JSON filename under output-dir.",
    )
    parser.add_argument(
        "--print-sample",
        type=int,
        default=1,
        help="Print the first N synthesized items to stdout.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.repeat_per_instance < 1:
        raise ValueError("--repeat-per-instance must be >= 1")

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        Path(args.cache_dir).expanduser()
        if args.cache_dir
        else Path.home() / ".codeminer"
    )

    # Load instances from file or dataset
    if args.selected_instances:
        selected_path = Path(args.selected_instances).expanduser()
        selected_instances = _load_json_list(selected_path)
        if args.instance_id:
            selected_instances = [
                row
                for row in selected_instances
                if row.get("instance_id") == args.instance_id
            ]
            if not selected_instances:
                raise ValueError(
                    f"instance_id {args.instance_id!r} not found in {selected_path}"
                )
    else:
        # Load directly from SWE-bench dataset
        dataset_name = (
            "princeton-nlp/SWE-bench_Lite"
            if args.dataset == "swebench_lite"
            else "princeton-nlp/SWE-bench_Verified"
        )
        filter_pattern = (
            f"^({args.instance_id})$" if args.instance_id else args.filter_instance
        )

        logger.info(
            "Loading instances from %s (split=%s, filter=%s)",
            dataset_name,
            args.split,
            filter_pattern,
        )

        dataset_obj = SwebenchDataset(
            dataset=dataset_name,
            split=args.split,
            filter_instance=filter_pattern,
            root=str(cache_dir),
            repo_root=args.repo_cache_dir or str(cache_dir),
        )
        dataset_instances = dataset_obj.load()

        # Convert to list of dicts
        selected_instances = [dict(instance) for instance in dataset_instances]

        if len(selected_instances) == 0:
            raise ValueError(
                f"No instances found in {dataset_name} with filter={filter_pattern}"
            )

        logger.info("Loaded %d instance(s) from dataset", len(selected_instances))

    allowed_tools = [
        tool.strip() for tool in args.allowed_tools.split(",") if tool.strip()
    ]
    query_types = _parse_query_types(args.query_types)

    limit = args.synthesis_limit or len(selected_instances)
    synth_inputs = selected_instances[:limit]
    missing_required = [
        row.get("instance_id", "unknown")
        for row in synth_inputs
        if not row.get("base_commit") or not row.get("repo")
    ]
    if missing_required:
        preview = ", ".join(missing_required[:5])
        raise ValueError(
            "selected_instances.json is missing required fields "
            "(base_commit/repo) for: "
            f"{preview} (total={len(missing_required)}). "
            "Please re-run scripts/collect_swebench.sh."
        )

    expanded_inputs: List[Dict[str, Any]] = []
    run_ids: List[int] = []
    for instance in synth_inputs:
        for run_id in range(1, args.repeat_per_instance + 1):
            expanded = dict(instance)
            expanded["synthesis_run_id"] = run_id
            expanded_inputs.append(expanded)
            run_ids.append(run_id)

    logger.info(
        "Synthesizing %d runs across %d instances, %d query types (repeat=%d).",
        len(expanded_inputs),
        len(synth_inputs),
        len(query_types),
        args.repeat_per_instance,
    )
    synthesized: List[Dict[str, Any]] = []
    for query_type in query_types:
        synthesizer = ClaudeQuerySynthesizer(
            model=args.model_name,
            max_turns=args.max_turns,
            allowed_tools=allowed_tools,
            permission_mode=args.permission_mode,
            query_type=query_type,
            sampling_seed=args.sampling_seed,
            behavioral_consensus_runs=args.behavioral_consensus_runs,
        )
        results = synthesizer.synthesize_queries(
            expanded_inputs,
            repo_root=args.repo_cache_dir,
            cache_dir=str(cache_dir),
        )

        if args.repeat_per_instance > 1:
            for result, run_id in zip(results, run_ids, strict=True):
                result["run_id"] = run_id
                if "query_id" in result:
                    result["query_id"] = f"{result['query_id']}_run{run_id}"

        synthesized.extend(results)

    synthesized = [_to_compact_record(item) for item in synthesized]

    output_path = output_dir / args.output_file
    _dump_json(synthesized, output_path)
    logger.info("Saved synthesized queries to %s", output_path)

    if args.print_sample:
        print(
            json.dumps(synthesized[: args.print_sample], indent=2, ensure_ascii=False)
        )


if __name__ == "__main__":
    main()
