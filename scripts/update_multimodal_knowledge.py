#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Update multimodal wiki knowledge and an optional visual vector sidecar."""

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
    DEFAULT_WEMM_MODEL,
    OpenAICompatibleVisualFactExtractor,
    WeMMVisualEmbeddingBackend,
    build_multimodal_repository_knowledge,
    build_visual_vector_index,
    load_visual_vector_index,
    save_multimodal_knowledge_bundle,
    save_visual_vector_index,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Repository root to scan")
    parser.add_argument(
        "--bundle-output",
        "-o",
        required=True,
        help="Path to write the multimodal knowledge bundle JSON",
    )
    parser.add_argument(
        "--visual-vector-output",
        help="Optional path to write the visual semantic vector sidecar JSON",
    )
    parser.add_argument(
        "--previous-visual-vector-index",
        help="Optional previous visual vector sidecar to reuse unchanged records",
    )
    parser.add_argument(
        "--visual-vector-backend",
        choices=("local", "wemm"),
        default="local",
        help="Embedding backend for the visual vector sidecar",
    )
    parser.add_argument(
        "--visual-vector-provider",
        default="local",
        help="Embedding provider label for the visual vector sidecar",
    )
    parser.add_argument(
        "--visual-vector-model",
        default=None,
        help="Embedding model; defaults according to --visual-vector-backend",
    )
    parser.add_argument(
        "--visual-vector-revision",
        default=None,
        help="Optional immutable model revision for the visual vector sidecar",
    )
    parser.add_argument(
        "--visual-vector-trust-remote-code",
        action="store_true",
        help=(
            "Allow remote model code. Remote models require a full 40-character "
            "commit SHA through --visual-vector-revision."
        ),
    )
    parser.add_argument(
        "--visual-vector-device",
        default=None,
        help="Optional SentenceTransformers device for the WeMM backend",
    )
    parser.add_argument(
        "--visual-vector-batch-size",
        type=int,
        default=1,
        help="Batch size for the WeMM backend",
    )
    parser.add_argument(
        "--visual-vector-dimensions",
        type=int,
        default=64,
        help="Visual vector sidecar dimensions",
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
        previous_vector_index = _load_previous_vector_index(args)
        bundle = build_multimodal_repository_knowledge(
            repo,
            commit=args.commit,
            exclude_roots=tuple(args.exclude_root),
            extractor=extractor,
            max_artifacts=args.max_artifacts,
            max_source_candidates=args.max_source_candidates,
        )
        save_multimodal_knowledge_bundle(bundle, args.bundle_output)
        counts = _bundle_counts(bundle)
        if args.visual_vector_output:
            vector_options = _visual_vector_options(args, repo_path=repo)
            vector_index = build_visual_vector_index(
                bundle["knowledge_view"],
                previous_index=previous_vector_index,
                **vector_options,
            )
            save_visual_vector_index(vector_index, args.visual_vector_output)
            counts.update(
                {
                    "visual_vector_records": vector_index["entry_count"],
                    "visual_vector_reused_records": vector_index["reused_record_count"],
                    "visual_vector_embedded_records": vector_index[
                        "embedded_record_count"
                    ],
                }
            )
    except ValueError as exc:
        parser.error(str(exc))
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
    return OpenAICompatibleVisualFactExtractor(
        model=model,
        api_base=api_base,
        api_key=os.environ.get(str(args.visual_facts_api_key_env or "")),
        timeout=args.visual_facts_timeout,
        provider=args.visual_facts_provider,
        repo_path=repo_path,
    )


def _load_previous_vector_index(args: argparse.Namespace) -> dict | None:
    if not args.previous_visual_vector_index:
        return None
    if not args.visual_vector_output:
        raise ValueError(
            "--previous-visual-vector-index requires --visual-vector-output"
        )
    return load_visual_vector_index(args.previous_visual_vector_index)


def _visual_vector_options(
    args: argparse.Namespace,
    *,
    repo_path: Path,
) -> dict:
    if args.visual_vector_backend == "local":
        if args.visual_vector_trust_remote_code:
            raise ValueError(
                "--visual-vector-trust-remote-code requires "
                "--visual-vector-backend wemm"
            )
        return {
            "provider": args.visual_vector_provider,
            "model": args.visual_vector_model or "local/hash-visual-embedding-v1",
            "model_revision": str(args.visual_vector_revision or ""),
            "dimensions": args.visual_vector_dimensions,
        }
    backend = WeMMVisualEmbeddingBackend(
        repo_path=repo_path,
        model=args.visual_vector_model or DEFAULT_WEMM_MODEL,
        dimensions=args.visual_vector_dimensions,
        revision=args.visual_vector_revision,
        trust_remote_code=args.visual_vector_trust_remote_code,
        device=args.visual_vector_device,
        batch_size=args.visual_vector_batch_size,
    )
    return {
        "document_embedder": backend.embed_documents,
        "provider": backend.provider,
        "model": backend.model_name,
        "model_revision": backend.model_revision,
        "dimensions": backend.dimensions,
        "document_modalities": backend.document_modalities,
        "query_modality": backend.query_modality,
    }


def _bundle_counts(bundle: dict) -> dict[str, int]:
    return {
        "media_artifacts": bundle["media_manifest"]["artifact_count"],
        "visual_fact_packs": bundle["visual_facts_manifest"]["fact_count"],
        "source_candidates": bundle["source_candidate_count"],
        "visual_code_bindings": bundle["grounding_manifest"]["binding_count"],
        "knowledge_entries": bundle["knowledge_view"]["entry_count"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
