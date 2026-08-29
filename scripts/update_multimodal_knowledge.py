#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Incrementally update multimodal knowledge and an optional vector sidecar."""

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
from codenib.wiki import (  # noqa: E402
    DEFAULT_WEMM_MODEL,
    OpenAICompatibleVisualFactExtractor,
    WeMMVisualEmbeddingBackend,
    build_index_backed_visual_grounding_scorer,
    build_multimodal_knowledge_bundle,
    build_multimodal_knowledge_view,
    build_multimodal_repository_knowledge,
    build_visual_facts_manifest,
    build_visual_vector_index,
    create_visual_vector_store,
    discover_media_manifest,
    discover_source_symbol_candidates,
    ground_visual_facts_to_sources,
    load_multimodal_knowledge_bundle,
    load_visual_vector_index,
    merge_incremental_visual_facts,
    plan_incremental_visual_fact_update,
    save_multimodal_knowledge_bundle,
    save_visual_vector_index,
    update_visual_vector_store,
)

_LOCAL_EXTRACTOR_ID = "local/metadata"
_MAX_EXTRACTOR_ID_BYTES = 128


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Repository root to scan")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--previous",
        help="Validated bundle to update; overwritten atomically unless --output is set",
    )
    mode.add_argument(
        "--bundle-output",
        help="Destination for an initial bundle when no previous bundle exists",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional destination for the updated bundle (default: --previous)",
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
    previous_path = Path(args.previous).expanduser() if args.previous else None
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else previous_path or Path(args.bundle_output).expanduser()
    )
    if not repo.exists():
        parser.error(f"repository root does not exist: {repo}")
    if not repo.is_dir():
        parser.error(f"repository root is not a directory: {repo}")
    repo = repo.resolve()
    try:
        extractor, extractor_id = _extractor(args, repo)
        scorer = _build_visual_grounding_scorer(args, repo_path=repo)
        if previous_path is None:
            if args.output:
                raise ValueError("--output requires --previous")
            if args.dry_run or args.force_reextract:
                raise ValueError("--dry-run and --force-reextract require --previous")
            updated = build_multimodal_repository_knowledge(
                repo,
                commit=args.commit,
                exclude_roots=tuple(args.exclude_root),
                extractor=extractor,
                scorer=scorer,
                max_artifacts=args.max_artifacts,
                max_source_candidates=args.max_source_candidates,
            )
            plan = None
            extracted_facts = None
        else:
            previous = load_multimodal_knowledge_bundle(previous_path)
            current_media = discover_media_manifest(
                repo,
                commit=args.commit,
                exclude_roots=tuple(args.exclude_root),
                selection=DEFAULT_REPOSITORY_SOURCE_SELECTION,
                max_artifacts=args.max_artifacts,
            )
            expected_id = (
                f"force/{extractor_id}" if args.force_reextract else extractor_id
            )
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
        (
            vector_index,
            query_embedder,
            previous_vector_index,
        ) = _build_visual_vector_sidecar(args, updated, repo_path=repo)
        save_multimodal_knowledge_bundle(updated, output_path)
        if vector_index is not None:
            save_visual_vector_index(vector_index, args.visual_vector_output)
            if args.visual_vector_store_output:
                store_stats = _materialize_visual_vector_store(
                    args,
                    vector_index=vector_index,
                    previous_vector_index=previous_vector_index,
                    query_embedder=query_embedder,
                )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    summary = (
        _update_summary(
            updated,
            plan,
            extracted_facts,
            extractor_id=extractor_id,
            grounding_backend=args.grounding_indexes,
            output_path=output_path,
        )
        if plan is not None and extracted_facts is not None
        else _initial_summary(
            updated,
            extractor_id=extractor_id,
            grounding_backend=args.grounding_indexes,
            output_path=output_path,
        )
    )
    if vector_index is not None:
        summary.update(
            {
                "visual_vector_records": vector_index["entry_count"],
                "visual_vector_reused_records": vector_index["reused_record_count"],
                "visual_vector_embedded_records": vector_index["embedded_record_count"],
            }
        )
        if args.visual_vector_store_output:
            summary.update(
                {
                    "visual_vector_store_records": store_stats["entry_count"],
                    "visual_vector_store_changed_records": store_stats[
                        "changed_entry_count"
                    ],
                    "visual_vector_store_update_mode": store_stats["mode"],
                }
            )
    print(json.dumps(summary, sort_keys=True))
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


def _build_visual_vector_sidecar(
    args: argparse.Namespace,
    bundle: dict,
    *,
    repo_path: Path,
) -> tuple[dict | None, object | None, dict | None]:
    _validate_visual_vector_store_options(args)
    if args.previous_visual_vector_index and not args.visual_vector_output:
        raise ValueError(
            "--previous-visual-vector-index requires --visual-vector-output"
        )
    if not args.visual_vector_output:
        return None, None, None
    previous = _load_previous_vector_index(args)
    vector_options, query_embedder = _visual_vector_options(
        args,
        repo_path=repo_path,
    )
    vector_index = build_visual_vector_index(
        bundle["knowledge_view"], previous_index=previous, **vector_options
    )
    return vector_index, query_embedder, previous


def _validate_visual_vector_store_options(args: argparse.Namespace) -> None:
    if args.visual_vector_store_output and not args.visual_vector_output:
        raise ValueError("--visual-vector-store-output requires --visual-vector-output")
    if args.previous_visual_vector_store and not args.visual_vector_store_output:
        raise ValueError(
            "--previous-visual-vector-store requires --visual-vector-store-output"
        )
    if args.previous_visual_vector_store and not args.previous_visual_vector_index:
        raise ValueError(
            "--previous-visual-vector-store requires --previous-visual-vector-index"
        )


def _load_previous_vector_index(args: argparse.Namespace) -> dict | None:
    _validate_visual_vector_store_options(args)
    if not args.previous_visual_vector_index:
        return None
    return load_visual_vector_index(args.previous_visual_vector_index)


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


def _initial_summary(
    bundle: dict,
    *,
    extractor_id: str,
    grounding_backend: str,
    output_path: Path,
) -> dict:
    return {
        "dry_run": False,
        "extractor": extractor_id,
        "grounding_backend": grounding_backend,
        "media_artifacts": bundle["media_manifest"]["artifact_count"],
        "visual_fact_packs": bundle["visual_facts_manifest"]["fact_count"],
        "visual_code_bindings": bundle["grounding_manifest"]["binding_count"],
        "knowledge_entries": bundle["knowledge_view"]["entry_count"],
        "bundle_sha256": bundle["bundle_sha256"],
        "output": os.fspath(output_path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
