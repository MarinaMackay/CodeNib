#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Incrementally update a persisted multimodal repository knowledge bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from codenib.repository_source_selection import DEFAULT_REPOSITORY_SOURCE_SELECTION
from codenib.wiki import (
    OpenAICompatibleVisualFactExtractor,
    OpenAICompatibleVisualGraphPlanExtractor,
    build_multimodal_knowledge_bundle,
    build_multimodal_knowledge_view,
    build_visual_facts_manifest,
    build_visual_graph_manifest,
    build_visual_storyboard_manifest,
    discover_media_manifest,
    discover_source_symbol_candidates,
    ground_visual_facts_to_sources,
    load_multimodal_knowledge_bundle,
    merge_incremental_visual_facts,
    plan_incremental_visual_fact_update,
    save_multimodal_knowledge_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Repository root to scan")
    parser.add_argument(
        "--previous",
        required=True,
        help="Previous codenib.multimodal-knowledge-bundle.v1 JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to write the updated bundle JSON",
    )
    parser.add_argument(
        "--commit",
        default=None,
        help="Optional commit identity to record in the current media manifest",
    )
    parser.add_argument(
        "--max-artifacts",
        type=int,
        default=4096,
        help="Maximum media artifacts to include",
    )
    parser.add_argument(
        "--max-source-candidates",
        type=int,
        default=8192,
        help="Maximum source-symbol candidates to consider for grounding",
    )
    parser.add_argument(
        "--visual-facts-model",
        default=None,
        help=(
            "Optional OpenAI-compatible VLM model for extracting changed visual "
            "facts. When omitted, deterministic local extraction is used."
        ),
    )
    parser.add_argument(
        "--visual-facts-api-base",
        default=None,
        help="OpenAI-compatible API base URL for --visual-facts-model",
    )
    parser.add_argument(
        "--visual-facts-api-key-env",
        default="CODENIB_WIKI_VISUAL_FACTS_API_KEY",
        help="Environment variable that contains the visual-facts API key",
    )
    parser.add_argument(
        "--visual-facts-provider",
        default="openai-compatible",
        help="Provider label recorded in extracted visual facts",
    )
    parser.add_argument(
        "--visual-facts-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for each visual-fact VLM request",
    )
    parser.add_argument(
        "--visual-plan-model",
        default=None,
        help="Optional OpenAI-compatible VLM model for planning visual graphs",
    )
    parser.add_argument(
        "--visual-plan-api-base",
        default=None,
        help="OpenAI-compatible API base URL for --visual-plan-model",
    )
    parser.add_argument(
        "--visual-plan-api-key-env",
        default="CODENIB_WIKI_VISUAL_GRAPH_API_KEY",
        help="Environment variable that contains the visual-plan API key",
    )
    parser.add_argument(
        "--visual-plan-provider",
        default="openai-compatible",
        help="Provider label recorded in VLM-planned graph metadata",
    )
    parser.add_argument(
        "--visual-plan-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for each visual graph planning request",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_path = Path(args.repo)
    previous_bundle = load_multimodal_knowledge_bundle(args.previous)
    current_media = discover_media_manifest(
        repo_path,
        commit=args.commit,
        selection=DEFAULT_REPOSITORY_SOURCE_SELECTION,
        max_artifacts=args.max_artifacts,
    )
    plan = plan_incremental_visual_fact_update(
        previous_bundle["media_manifest"],
        current_media,
        previous_bundle["visual_facts_manifest"],
    )
    current_extract_media = _media_manifest_subset(
        current_media,
        set(plan["extract_artifact_paths"]),
    )
    extractor = _build_visual_fact_extractor(args, repo_path)
    facts_kwargs = {"extractor": extractor} if extractor is not None else {}
    new_facts = build_visual_facts_manifest(current_extract_media, **facts_kwargs)
    visual_facts = merge_incremental_visual_facts(
        current_media,
        reusable_fact_packs=plan["reusable_fact_packs"],
        new_fact_packs=new_facts["facts"],
    )
    source_candidates = discover_source_symbol_candidates(
        repo_path,
        selection=DEFAULT_REPOSITORY_SOURCE_SELECTION,
        max_candidates=args.max_source_candidates,
    )
    grounding = ground_visual_facts_to_sources(visual_facts, source_candidates)
    knowledge_view = build_multimodal_knowledge_view(
        current_media,
        visual_facts,
        grounding,
    )
    graph_planner = _build_visual_graph_planner(args, repo_path)
    visual_graph_manifest = build_visual_graph_manifest(
        knowledge_view,
        planner=graph_planner,
    )
    visual_storyboard_manifest = build_visual_storyboard_manifest(visual_graph_manifest)
    bundle = build_multimodal_knowledge_bundle(
        media_manifest=current_media,
        visual_facts_manifest=visual_facts,
        source_candidate_count=len(source_candidates),
        grounding_manifest=grounding,
        knowledge_view=knowledge_view,
        incremental_update=plan,
        visual_graph_manifest=visual_graph_manifest,
        visual_storyboard_manifest=visual_storyboard_manifest,
    )
    save_multimodal_knowledge_bundle(bundle, args.output)
    print(json.dumps(_summary(bundle, plan, new_facts), sort_keys=True))
    return 0


def _media_manifest_subset(media_manifest: dict, paths: set[str]) -> dict:
    subset = dict(media_manifest)
    subset["artifacts"] = [
        artifact
        for artifact in media_manifest.get("artifacts") or ()
        if artifact.get("path") in paths
    ]
    subset["artifact_count"] = len(subset["artifacts"])
    return subset


def _build_visual_fact_extractor(args: argparse.Namespace, repo_path: Path):
    model = str(args.visual_facts_model or "").strip()
    api_base = str(args.visual_facts_api_base or "").strip()
    if not model and not api_base:
        return None
    extractor = OpenAICompatibleVisualFactExtractor(
        model=model,
        api_base=api_base,
        api_key=os.environ.get(str(args.visual_facts_api_key_env or "")),
        timeout=args.visual_facts_timeout,
        provider=args.visual_facts_provider,
    )
    return lambda artifact: extractor.extract(artifact, repo_path=repo_path)


def _build_visual_graph_planner(args: argparse.Namespace, repo_path: Path):
    model = str(args.visual_plan_model or "").strip()
    api_base = str(args.visual_plan_api_base or "").strip()
    if not model and not api_base:
        return None
    planner = OpenAICompatibleVisualGraphPlanExtractor(
        model=model,
        api_base=api_base,
        api_key=os.environ.get(str(args.visual_plan_api_key_env or "")),
        timeout=args.visual_plan_timeout,
        provider=args.visual_plan_provider,
    )
    return lambda entry: planner.plan(entry, repo_path=repo_path)


def _summary(bundle: dict, plan: dict, new_facts: dict) -> dict:
    counts = dict((plan.get("media_diff") or {}).get("counts") or {})
    return {
        **counts,
        "reused_visual_fact_packs": len(plan.get("reusable_fact_packs") or ()),
        "regenerated_visual_fact_packs": new_facts.get("fact_count", 0),
        "visual_fact_packs": bundle["visual_facts_manifest"]["fact_count"],
        "visual_code_bindings": bundle["grounding_manifest"]["binding_count"],
        "visual_graph_plans": bundle["visual_graph_manifest"]["plan_count"],
        "visual_storyboards": bundle["visual_storyboard_manifest"]["storyboard_count"],
        "knowledge_entries": bundle["knowledge_view"]["entry_count"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
