#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Build a deterministic multimodal repository knowledge bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from codenib.wiki import (  # noqa: E402
    IndexBackedVisualGroundingScorer,
    OpenAICompatibleVisualFactExtractor,
    build_multimodal_repository_knowledge,
    save_multimodal_knowledge_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Repository root to scan")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to write the multimodal knowledge bundle JSON",
    )
    parser.add_argument(
        "--commit",
        default=None,
        help="Optional commit identity to record in the media manifest",
    )
    parser.add_argument(
        "--exclude-root",
        action="append",
        default=[],
        help="Path to exclude from repository discovery; may be repeated",
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
        "--grounding-indexes",
        choices=("lexical", "bm25", "bm25+lsp"),
        default="lexical",
        help="Grounding evidence backend (default: deterministic lexical)",
    )
    parser.add_argument(
        "--grounding-cache-dir",
        default=None,
        help="Optional CodeNib index cache used by index-backed grounding",
    )
    parser.add_argument(
        "--grounding-language",
        action="append",
        default=[],
        help="Language to index for grounding; may be repeated (default: python)",
    )
    parser.add_argument(
        "--visual-facts-model",
        default=None,
        help=(
            "Optional OpenAI-compatible VLM model for extracting visual facts. "
            "When omitted, deterministic local extraction is used."
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo = Path(args.repo).expanduser()
    if not repo.exists():
        parser.error(f"repository root does not exist: {repo}")
    if not repo.is_dir():
        parser.error(f"repository root is not a directory: {repo}")
    try:
        extractor = _build_visual_fact_extractor(args, repo_path=repo)
        scorer = _build_visual_grounding_scorer(args, repo_path=repo)
    except ValueError as exc:
        parser.error(str(exc))
    bundle = build_multimodal_repository_knowledge(
        repo,
        commit=args.commit,
        exclude_roots=tuple(args.exclude_root),
        extractor=extractor,
        scorer=scorer,
        max_artifacts=args.max_artifacts,
        max_source_candidates=args.max_source_candidates,
    )
    save_multimodal_knowledge_bundle(bundle, args.output)
    counts = {
        "grounding_backend": args.grounding_indexes,
        "media_artifacts": bundle["media_manifest"]["artifact_count"],
        "visual_fact_packs": bundle["visual_facts_manifest"]["fact_count"],
        "source_candidates": bundle["source_candidate_count"],
        "visual_code_bindings": bundle["grounding_manifest"]["binding_count"],
        "knowledge_entries": bundle["knowledge_view"]["entry_count"],
    }
    print(json.dumps(counts, sort_keys=True))
    return 0


def _build_visual_fact_extractor(
    args: argparse.Namespace,
    *,
    repo_path: Path,
) -> OpenAICompatibleVisualFactExtractor | None:
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
        repo_path=repo_path,
    )
    return extractor


def _build_visual_grounding_scorer(
    args: argparse.Namespace,
    *,
    repo_path: Path,
) -> IndexBackedVisualGroundingScorer | None:
    mode = str(args.grounding_indexes or "lexical")
    if mode == "lexical":
        return None
    from codenib.compiler.skill_context import build_skill_contexts

    skill_ids = ["bm25_search"]
    if mode == "bm25+lsp":
        skill_ids.extend(["lsp_definition", "lsp_references"])
    contexts = build_skill_contexts(
        str(repo_path),
        skill_ids,
        languages=tuple(args.grounding_language or ("python",)),
        cache_dir=args.grounding_cache_dir,
        skills_dir=str(Path(_PROJECT_ROOT) / "codenib" / "agent" / "skills"),
    )
    retrieve = contexts.get("retrieve")
    bm25 = getattr(retrieve, "bm25", None)
    expand = contexts.get("expand")
    lsp_provider = getattr(expand, "lsp_provider", None)
    if mode == "bm25+lsp" and lsp_provider is None:
        code_graph = getattr(expand, "code_graph", None)
        if code_graph is not None:
            from codenib.agent.lsp_provider import StaticLSPProvider

            lsp_provider = StaticLSPProvider(code_graph)
    if bm25 is None:
        raise ValueError("index-backed grounding did not load a BM25 index")
    if mode == "bm25+lsp" and lsp_provider is None:
        raise ValueError("bm25+lsp grounding did not load an LSP provider")
    return IndexBackedVisualGroundingScorer(
        bm25=bm25,
        lsp_provider=lsp_provider if mode == "bm25+lsp" else None,
        repo_path=repo_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
