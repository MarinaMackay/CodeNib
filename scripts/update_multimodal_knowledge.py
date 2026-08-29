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
    create_visual_vector_store,
    load_visual_vector_index,
    save_multimodal_knowledge_bundle,
    save_visual_vector_index,
    update_visual_vector_store,
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
        "--visual-vector-store-output",
        help="Optional directory to materialize a searchable FAISS visual store",
    )
    parser.add_argument(
        "--previous-visual-vector-store",
        help="Optional previous FAISS visual store to update incrementally",
    )
    parser.add_argument(
        "--visual-vector-store-index-type",
        choices=("flat", "ivf"),
        default="flat",
        help="FAISS index type for the visual vector store",
    )
    parser.add_argument(
        "--visual-vector-delta-threshold",
        type=float,
        default=0.1,
        help="Maximum changed-entry ratio for an in-place flat-index update",
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
            vector_options, query_embedder = _visual_vector_options(
                args,
                repo_path=repo,
            )
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
            if args.visual_vector_store_output:
                store_stats = _materialize_visual_vector_store(
                    args,
                    vector_index=vector_index,
                    previous_vector_index=previous_vector_index,
                    query_embedder=query_embedder,
                )
                counts.update(
                    {
                        "visual_vector_store_records": store_stats["entry_count"],
                        "visual_vector_store_changed_records": store_stats[
                            "changed_entry_count"
                        ],
                        "visual_vector_store_update_mode": store_stats["mode"],
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
    if args.visual_vector_store_output and not args.visual_vector_output:
        raise ValueError("--visual-vector-store-output requires --visual-vector-output")
    if args.previous_visual_vector_store and not args.visual_vector_store_output:
        raise ValueError(
            "--previous-visual-vector-store requires " "--visual-vector-store-output"
        )
    if args.previous_visual_vector_store and not args.previous_visual_vector_index:
        raise ValueError(
            "--previous-visual-vector-store requires " "--previous-visual-vector-index"
        )
    if not args.previous_visual_vector_index:
        return None
    if not args.visual_vector_output:
        raise ValueError(
            "--previous-visual-vector-index requires --visual-vector-output"
        )
    return load_visual_vector_index(args.previous_visual_vector_index)


def _materialize_visual_vector_store(
    args: argparse.Namespace,
    *,
    vector_index: dict,
    previous_vector_index: dict | None,
    query_embedder=None,
) -> dict:
    output = Path(args.visual_vector_store_output).expanduser().resolve()
    previous = (
        Path(args.previous_visual_vector_store).expanduser().resolve()
        if args.previous_visual_vector_store
        else None
    )
    if previous is not None:
        if not previous.is_dir():
            raise ValueError(
                f"previous visual vector store is not a directory: {previous}"
            )
        if (
            previous_vector_index is None
            or previous_vector_index["embedding_policy_sha256"]
            != vector_index["embedding_policy_sha256"]
        ):
            raise ValueError(
                "previous visual vector store embedding policy does not match"
            )
    same_store = previous is not None and output == previous
    if output.exists() and not output.is_dir():
        raise ValueError(f"visual vector store output is not a directory: {output}")
    if output.is_dir() and any(output.iterdir()) and not same_store:
        raise ValueError(
            "visual vector store output must be empty unless it is the previous store"
        )

    store = create_visual_vector_store(
        vector_index,
        query_embedder=query_embedder,
        store_path=output,
        index_type=args.visual_vector_store_index_type,
    )
    try:
        if previous is not None:
            _load_trusted_local_visual_store(store, previous)
        stats = update_visual_vector_store(
            store,
            vector_index,
            previous_index=previous_vector_index if previous is not None else None,
            threshold=args.visual_vector_delta_threshold,
        )
        store.save(output)
        return stats
    finally:
        store.close()


def _load_trusted_local_visual_store(store, path: Path) -> None:
    from codenib.index.embedding.artifact_integrity import (
        capture_authenticated_vector_view,
    )
    from codenib.native_index_authorization import (
        _mint_trusted_local_admin_authorization,
    )

    with capture_authenticated_vector_view(path) as view:
        authorization = _mint_trusted_local_admin_authorization(
            view.ownership,
            view_type="vector",
            semantic_contract=store.artifact_metadata,
            evidence=(
                "update-multimodal-knowledge-local-cli",
                "explicit-previous-visual-vector-store",
            ),
        )
    store.load(path, native_index_authorization=authorization)


def _visual_vector_options(
    args: argparse.Namespace,
    *,
    repo_path: Path,
) -> tuple[dict, object | None]:
    if args.visual_vector_backend == "local":
        if args.visual_vector_trust_remote_code:
            raise ValueError(
                "--visual-vector-trust-remote-code requires "
                "--visual-vector-backend wemm"
            )
        return (
            {
                "provider": args.visual_vector_provider,
                "model": args.visual_vector_model or "local/hash-visual-embedding-v1",
                "model_revision": str(args.visual_vector_revision or ""),
                "dimensions": args.visual_vector_dimensions,
            },
            None,
        )
    backend = WeMMVisualEmbeddingBackend(
        repo_path=repo_path,
        model=args.visual_vector_model or DEFAULT_WEMM_MODEL,
        dimensions=args.visual_vector_dimensions,
        revision=args.visual_vector_revision,
        trust_remote_code=args.visual_vector_trust_remote_code,
        device=args.visual_vector_device,
        batch_size=args.visual_vector_batch_size,
    )
    return (
        {
            "document_embedder": backend.embed_documents,
            "provider": backend.provider,
            "model": backend.model_name,
            "model_revision": backend.model_revision,
            "dimensions": backend.dimensions,
            "document_modalities": backend.document_modalities,
            "query_modality": backend.query_modality,
        },
        backend.embed_queries,
    )


def _bundle_counts(bundle: dict) -> dict:
    return {
        "media_artifacts": bundle["media_manifest"]["artifact_count"],
        "visual_fact_packs": bundle["visual_facts_manifest"]["fact_count"],
        "source_candidates": bundle["source_candidate_count"],
        "visual_code_bindings": bundle["grounding_manifest"]["binding_count"],
        "knowledge_entries": bundle["knowledge_view"]["entry_count"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
