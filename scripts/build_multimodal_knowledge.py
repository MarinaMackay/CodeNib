#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Build a deterministic multimodal repository knowledge bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from codenib.wiki import IndexBackedVisualGroundingScorer  # noqa: E402
from codenib.wiki import (
    OpenAICompatibleVisualFactExtractor,
    build_index_backed_visual_grounding_scorer,
    build_multimodal_repository_knowledge,
    build_visual_graph_manifest,
    build_visual_storyboard_manifest,
    compile_visual_graph_plan_to_archify,
    compile_visual_graph_plan_to_mermaid,
    compile_visual_storyboard_to_markdown,
    render_visual_storyboard_manifest_videos,
    save_archify_architecture,
    save_multimodal_knowledge_bundle,
    save_visual_graph_manifest,
    save_visual_storyboard_manifest,
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
    parser.add_argument(
        "--visual-graph-output",
        default=None,
        help="Optional path to write a validated visual graph manifest JSON",
    )
    parser.add_argument(
        "--visual-graph-mermaid-dir",
        default=None,
        help="Optional directory to write one compiled Mermaid file per plan",
    )
    parser.add_argument(
        "--visual-storyboard-output",
        default=None,
        help="Optional path to write a validated video-ready storyboard manifest",
    )
    parser.add_argument(
        "--visual-storyboard-markdown-dir",
        default=None,
        help="Optional directory to write one inspectable Markdown shot list per asset",
    )
    parser.add_argument(
        "--visual-storyboard-video-dir",
        default=None,
        help="Optional directory to render real local MP4 videos with ffmpeg",
    )
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="Optional absolute path to the ffmpeg executable",
    )
    parser.add_argument(
        "--visual-graph-archify-dir",
        default=None,
        help="Optional directory to write Archify architecture JSON per plan",
    )
    parser.add_argument(
        "--archify-repository-url",
        default=None,
        help="Public GitHub repository URL for revision-pinned Archify evidence",
    )
    parser.add_argument(
        "--archify-revision",
        default=None,
        help="Full repository commit SHA for revision-pinned Archify evidence",
    )
    parser.add_argument(
        "--archify-min-grounding-score",
        type=float,
        default=0.8,
        help="Minimum grounding score exported as Archify source evidence",
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
    visual_graph_manifest = None
    if (
        args.visual_graph_output
        or args.visual_graph_mermaid_dir
        or args.visual_storyboard_output
        or args.visual_storyboard_markdown_dir
        or args.visual_storyboard_video_dir
        or args.visual_graph_archify_dir
    ):
        visual_graph_manifest = build_visual_graph_manifest(bundle["knowledge_view"])
    if args.visual_graph_output:
        save_visual_graph_manifest(visual_graph_manifest, args.visual_graph_output)
    if args.visual_graph_mermaid_dir:
        _write_mermaid_plans(visual_graph_manifest, args.visual_graph_mermaid_dir)
    visual_storyboard_manifest = None
    if (
        args.visual_storyboard_output
        or args.visual_storyboard_markdown_dir
        or args.visual_storyboard_video_dir
    ):
        visual_storyboard_manifest = build_visual_storyboard_manifest(
            visual_graph_manifest
        )
    if args.visual_storyboard_output:
        save_visual_storyboard_manifest(
            visual_storyboard_manifest, args.visual_storyboard_output
        )
    if args.visual_storyboard_markdown_dir:
        _write_storyboards(
            visual_storyboard_manifest, args.visual_storyboard_markdown_dir
        )
    video_manifest = None
    if args.visual_storyboard_video_dir:
        try:
            video_manifest = render_visual_storyboard_manifest_videos(
                visual_storyboard_manifest,
                args.visual_storyboard_video_dir,
                ffmpeg=args.ffmpeg,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.visual_graph_archify_dir:
        try:
            _write_archify_plans(visual_graph_manifest, args)
        except ValueError as exc:
            parser.error(str(exc))
    counts = {
        "grounding_backend": args.grounding_indexes,
        "media_artifacts": bundle["media_manifest"]["artifact_count"],
        "visual_fact_packs": bundle["visual_facts_manifest"]["fact_count"],
        "source_candidates": bundle["source_candidate_count"],
        "visual_code_bindings": bundle["grounding_manifest"]["binding_count"],
        "knowledge_entries": bundle["knowledge_view"]["entry_count"],
    }
    if visual_graph_manifest is not None:
        counts["visual_graph_plans"] = visual_graph_manifest["plan_count"]
    if visual_storyboard_manifest is not None:
        counts["visual_storyboards"] = visual_storyboard_manifest["storyboard_count"]
    if video_manifest is not None:
        counts["storyboard_videos"] = video_manifest["video_count"]
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
    return build_index_backed_visual_grounding_scorer(
        repo_path,
        mode=mode,
        languages=tuple(args.grounding_language or ("python",)),
        cache_dir=args.grounding_cache_dir,
    )


def _write_mermaid_plans(manifest: dict, output_dir: str | Path) -> None:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    for index, plan in enumerate(manifest["plans"], start=1):
        target = destination / _mermaid_filename(plan["artifact_path"], index)
        _atomic_write_text(target, compile_visual_graph_plan_to_mermaid(plan))


def _write_storyboards(manifest: dict, output_dir: str | Path) -> None:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    for index, storyboard in enumerate(manifest["storyboards"], start=1):
        filename = _artifact_filename(storyboard["artifact_path"], index, ".md")
        _atomic_write_text(
            destination / filename,
            compile_visual_storyboard_to_markdown(storyboard),
        )


def _write_archify_plans(manifest: dict, args: argparse.Namespace) -> None:
    destination = Path(args.visual_graph_archify_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    for index, plan in enumerate(manifest["plans"], start=1):
        document = compile_visual_graph_plan_to_archify(
            plan,
            repository_url=args.archify_repository_url,
            revision=args.archify_revision,
            minimum_grounding_score=args.archify_min_grounding_score,
        )
        target = destination / _archify_filename(plan["artifact_path"], index)
        save_archify_architecture(document, target)


def _mermaid_filename(artifact_path: str, index: int) -> str:
    return _artifact_filename(artifact_path, index, ".mmd")


def _artifact_filename(artifact_path: str, index: int, suffix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", artifact_path).strip(".-")
    stem = stem[:80].rstrip(".-") or "artifact"
    digest = hashlib.sha256(artifact_path.encode("utf-8")).hexdigest()[:12]
    return f"{index:04d}-{stem}-{digest}{suffix}"


def _archify_filename(artifact_path: str, index: int) -> str:
    return (
        _mermaid_filename(artifact_path, index).removesuffix(".mmd")
        + ".architecture.json"
    )


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.encode("utf-8")
    try:
        existing = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        existing_mode = None
    else:
        existing_mode = (
            stat.S_IMODE(existing.st_mode) if stat.S_ISREG(existing.st_mode) else None
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            if existing_mode is not None:
                os.fchmod(stream.fileno(), existing_mode)
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
