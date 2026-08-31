#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Incrementally update a persisted multimodal repository knowledge bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from codenib.repository_source_selection import (  # noqa: E402
    DEFAULT_REPOSITORY_SOURCE_SELECTION,
)
from codenib.wiki import IndexBackedVisualGroundingScorer  # noqa: E402
from codenib.wiki import (
    OpenAICompatibleVisualFactExtractor,
    build_index_backed_visual_grounding_scorer,
    build_multimodal_knowledge_bundle,
    build_multimodal_knowledge_view,
    build_visual_facts_manifest,
    discover_media_manifest,
    discover_source_symbol_candidates,
    ground_visual_facts_to_sources,
    load_multimodal_knowledge_bundle,
    merge_incremental_visual_facts,
    plan_incremental_visual_fact_update,
    save_multimodal_knowledge_bundle,
)

_LOCAL_EXTRACTOR_ID = "local/metadata"
_MAX_EXTRACTOR_ID_BYTES = 128


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Repository root to scan")
    parser.add_argument(
        "--previous",
        required=True,
        help="Validated bundle to update; overwritten atomically unless --output is set",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional destination for the updated bundle (default: --previous)",
    )
    parser.add_argument(
        "--commit",
        default=None,
        help="Optional commit identity to record in the current media manifest",
    )
    parser.add_argument(
        "--exclude-root",
        action="append",
        default=[],
        help="Path to exclude from discovery; may be repeated",
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
        "--dry-run",
        action="store_true",
        help="Print the incremental plan without extraction, grounding, or writes",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Regenerate every current visual fact pack under the selected extractor",
    )
    parser.add_argument(
        "--visual-facts-model",
        default=None,
        help="Optional OpenAI-compatible VLM model for changed visual artifacts",
    )
    parser.add_argument(
        "--visual-facts-api-base",
        default=None,
        help="OpenAI-compatible API base URL for --visual-facts-model",
    )
    parser.add_argument(
        "--visual-facts-api-key-env",
        default="CODENIB_WIKI_VISUAL_FACTS_API_KEY",
        help="Environment variable containing the visual-facts API key",
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
        help="Timeout in seconds for each changed-artifact VLM request",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo = Path(args.repo).expanduser()
    previous_path = Path(args.previous).expanduser()
    output_path = Path(args.output).expanduser() if args.output else previous_path
    if not repo.exists():
        parser.error(f"repository root does not exist: {repo}")
    if not repo.is_dir():
        parser.error(f"repository root is not a directory: {repo}")
    repo = repo.resolve()
    try:
        previous = load_multimodal_knowledge_bundle(previous_path)
        current_media = discover_media_manifest(
            repo,
            commit=args.commit,
            exclude_roots=tuple(args.exclude_root),
            selection=DEFAULT_REPOSITORY_SOURCE_SELECTION,
            max_artifacts=args.max_artifacts,
        )
        extractor, extractor_id = _extractor(args, repo)
        expected_id = f"force/{extractor_id}" if args.force_reextract else extractor_id
        plan = plan_incremental_visual_fact_update(
            previous["media_manifest"],
            current_media,
            previous["visual_facts_manifest"],
            expected_extractor=expected_id,
        )
        if args.dry_run:
            print(json.dumps(_plan_summary(plan, extractor_id), sort_keys=True))
            return 0
        extract_media = _media_manifest_subset(
            current_media, set(plan["extract_artifact_paths"])
        )
        facts_kwargs = {"extractor": extractor} if extractor is not None else {}
        extracted_facts = build_visual_facts_manifest(extract_media, **facts_kwargs)
        visual_facts = merge_incremental_visual_facts(
            current_media,
            reusable_fact_packs=plan["reusable_fact_packs"],
            new_fact_packs=extracted_facts["facts"],
        )
        scorer = _build_visual_grounding_scorer(args, repo_path=repo)
        source_candidates = discover_source_symbol_candidates(
            repo,
            exclude_roots=tuple(args.exclude_root),
            selection=DEFAULT_REPOSITORY_SOURCE_SELECTION,
            max_candidates=args.max_source_candidates,
        )
        augment_candidates = getattr(scorer, "augment_source_candidates", None)
        if callable(augment_candidates):
            source_candidates = augment_candidates(
                visual_facts,
                source_candidates,
                max_candidates=args.max_source_candidates,
            )
        grounding = ground_visual_facts_to_sources(
            visual_facts,
            source_candidates,
            scorer=scorer,
        )
        knowledge_view = build_multimodal_knowledge_view(
            current_media, visual_facts, grounding
        )
        updated = build_multimodal_knowledge_bundle(
            media_manifest=current_media,
            visual_facts_manifest=visual_facts,
            source_candidate_count=len(source_candidates),
            grounding_manifest=grounding,
            knowledge_view=knowledge_view,
        )
        save_multimodal_knowledge_bundle(updated, output_path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            _update_summary(
                updated,
                plan,
                extracted_facts,
                extractor_id=extractor_id,
                grounding_backend=args.grounding_indexes,
                output_path=output_path,
            ),
            sort_keys=True,
        )
    )
    return 0


def _extractor(
    args: argparse.Namespace, repo: Path
) -> tuple[OpenAICompatibleVisualFactExtractor | None, str]:
    model = str(args.visual_facts_model or "").strip()
    api_base = str(args.visual_facts_api_base or "").strip()
    if bool(model) != bool(api_base):
        raise ValueError(
            "--visual-facts-model and --visual-facts-api-base must be set together"
        )
    if not model:
        return None, _LOCAL_EXTRACTOR_ID
    provider = str(args.visual_facts_provider or "").strip()
    identity = _extractor_identity(provider, model)
    return (
        OpenAICompatibleVisualFactExtractor(
            model=model,
            api_base=api_base,
            api_key=os.environ.get(str(args.visual_facts_api_key_env or "")),
            timeout=args.visual_facts_timeout,
            provider=identity,
            repo_path=repo,
        ),
        identity,
    )


def _build_visual_grounding_scorer(
    args: argparse.Namespace,
    *,
    repo_path: Path,
) -> IndexBackedVisualGroundingScorer | None:
    mode = str(args.grounding_indexes or "lexical")
    if mode == "lexical":
        return None
    return build_index_backed_visual_grounding_scorer(
        repo_path,
        mode=mode,
        languages=tuple(args.grounding_language or ("python",)),
        cache_dir=args.grounding_cache_dir,
    )


def _extractor_identity(provider: str, model: str) -> str:
    provider = provider.strip()
    model = model.strip()
    if not provider or not model:
        raise ValueError("visual fact provider and model identity are required")
    readable = f"{provider}/{model}"
    if len(readable.encode("utf-8")) <= _MAX_EXTRACTOR_ID_BYTES:
        return readable
    digest = hashlib.sha256(readable.encode("utf-8")).hexdigest()[:20]
    budget = _MAX_EXTRACTOR_ID_BYTES - len(f"/sha256:{digest}".encode("utf-8"))
    prefix = provider.encode("utf-8")[:budget].decode("utf-8", errors="ignore").rstrip()
    if not prefix:
        prefix = "provider"
    return f"{prefix}/sha256:{digest}"


def _media_manifest_subset(media_manifest: dict, paths: set[str]) -> dict:
    subset = dict(media_manifest)
    subset["artifacts"] = [
        artifact
        for artifact in media_manifest["artifacts"]
        if artifact["path"] in paths
    ]
    subset["artifact_count"] = len(subset["artifacts"])
    return subset


def _plan_summary(plan: dict, extractor_id: str) -> dict:
    return {
        **plan["media_diff"]["counts"],
        "dry_run": True,
        "extractor": extractor_id,
        "extract_artifacts": len(plan["extract_artifact_paths"]),
        "reuse_visual_fact_packs": len(plan["reusable_fact_packs"]),
        "removed_artifacts": len(plan["removed_artifact_paths"]),
        "plan_sha256": plan["plan_sha256"],
    }


def _update_summary(
    bundle: dict,
    plan: dict,
    extracted_facts: dict,
    *,
    extractor_id: str,
    grounding_backend: str,
    output_path: Path,
) -> dict:
    return {
        **plan["media_diff"]["counts"],
        "dry_run": False,
        "extractor": extractor_id,
        "grounding_backend": grounding_backend,
        "reused_visual_fact_packs": len(plan["reusable_fact_packs"]),
        "regenerated_visual_fact_packs": extracted_facts["fact_count"],
        "visual_fact_packs": bundle["visual_facts_manifest"]["fact_count"],
        "visual_code_bindings": bundle["grounding_manifest"]["binding_count"],
        "knowledge_entries": bundle["knowledge_view"]["entry_count"],
        "bundle_sha256": bundle["bundle_sha256"],
        "output": os.fspath(output_path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
