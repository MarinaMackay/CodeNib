#!/usr/bin/env python3
"""
Collect SWE-bench samples and optionally synthesize natural-language queries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

from codeminer.dataset.collect.swebench_sample import SamplingConfig, run_sampling
from codeminer.dataset.synthesize import ClaudeQuerySynthesizer
from codeminer.log_utils import get_logger

logger = get_logger(__name__)


def _parse_languages(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not values:
        return None
    languages: List[str] = []
    for val in values:
        languages.extend([v for v in val.split(",") if v])
    return {lang.strip() for lang in languages if lang.strip()}


def _dump_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample SWE-bench instances and synthesize natural-language queries."
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["collect", "synthesize", "all"],
        default="all",
        help="Run only a specific stage or both (default: all).",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Languages to include (space-separated or comma-separated).",
    )
    parser.add_argument(
        "--multilingual-csv-path",
        type=str,
        default=None,
        help="Path to swebench_multilingual.csv when sampling non-Python languages.",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split.")
    parser.add_argument(
        "--filter-instance",
        type=str,
        default=".*",
        help="Regex filter for instance IDs.",
    )
    parser.add_argument("--repos-per-language", type=int, default=5)
    parser.add_argument("--instances-per-repo", type=int, default=3)
    parser.add_argument("--min-instances", type=int, default=3)
    parser.add_argument("--shallow-clone", action="store_true", default=True)
    parser.add_argument(
        "--no-shallow-clone", action="store_false", dest="shallow_clone"
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--repo-cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--print-sample",
        type=int,
        default=5,
        help="Print the first N sampled instances to stdout.",
    )
    parser.add_argument(
        "--save-sampled",
        type=str,
        default=None,
        help="Optional path to save sampled instances as JSON.",
    )

    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Generate natural-language queries using an LLM.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sonnet",
        help="Claude agent model name (e.g., sonnet, opus).",
    )
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument(
        "--permission-mode",
        type=str,
        default="bypassPermissions",
        help="Claude agent permission mode.",
    )
    parser.add_argument(
        "--allowed-tools",
        type=str,
        default="",
        help="Comma-separated list of allowed tools (optional).",
    )
    parser.add_argument(
        "--synthesis-limit",
        type=int,
        default=None,
        help="Limit the number of instances to synthesize.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    languages = _parse_languages(args.languages)

    config = SamplingConfig(
        languages=languages or SamplingConfig().languages,
        shallow_clone=args.shallow_clone,
        min_instances=args.min_instances,
        repos_per_language=args.repos_per_language,
        instances_per_repo=args.instances_per_repo,
        dataset_split=args.split,
        filter_instance=args.filter_instance,
        multilingual_csv_path=(
            Path(args.multilingual_csv_path) if args.multilingual_csv_path else None
        ),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        repo_cache_dir=Path(args.repo_cache_dir) if args.repo_cache_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    results = run_sampling(config)
    selected_instances = results.selected_instances

    if args.print_sample:
        preview = selected_instances[: args.print_sample]
        logger.info("Sampled %d instances. Preview:", len(selected_instances))
        print(json.dumps(preview, indent=2, ensure_ascii=False))

    if args.save_sampled:
        _dump_json(selected_instances, Path(args.save_sampled))

    if args.stage == "collect" or not args.synthesize:
        return

    allowed_tools = [
        tool.strip() for tool in args.allowed_tools.split(",") if tool.strip()
    ]
    synthesizer = ClaudeQuerySynthesizer(
        model=args.model_name,
        max_turns=args.max_turns,
        allowed_tools=allowed_tools,
        permission_mode=args.permission_mode,
    )

    limit = args.synthesis_limit or len(selected_instances)
    synth_inputs = selected_instances[:limit]
    logger.info("Synthesizing queries for %d instances.", len(synth_inputs))
    synthesized = synthesizer.synthesize_queries(synth_inputs)

    output_dir = results.output_dir or Path(args.output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(synthesized, output_dir / "synthesized_queries.json")

    logger.info(
        "Saved synthesized queries to %s", output_dir / "synthesized_queries.json"
    )


if __name__ == "__main__":
    main()
