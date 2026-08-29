# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from codenib.wiki import (
    build_visual_vector_index,
    load_visual_vector_index,
    save_visual_vector_index,
    search_visual_vector_index,
    validate_visual_vector_index,
)
from scripts.update_multimodal_knowledge import _visual_vector_options, build_parser
from scripts.update_multimodal_knowledge import main as update_multimodal_knowledge


def _sha256_json(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _knowledge_view():
    entries = [
        {
            "artifact": {
                "path": "docs/architecture.svg",
                "sha256": "a" * 64,
                "mime_type": "image/svg+xml",
                "caption": "Request router architecture",
                "role_hint": "architecture_diagram",
                "surrounding_text": "The request router renders CodeWiki pages.",
            },
            "facts": {
                "entities": [{"name": "WikiServer", "type": "component"}],
                "claims": [
                    {
                        "text": "The router dispatches requests to WikiServer.",
                        "evidence": "router arrow in the diagram",
                    }
                ],
            },
            "bindings": [
                {
                    "source_path": "codenib/web/app.py",
                    "symbol": "create_app",
                    "entity_name": "WikiServer",
                    "evidence": "exact symbol match",
                }
            ],
            "search_text": "request router architecture WikiServer create_app",
        },
        {
            "artifact": {
                "path": "docs/embedding-cache.svg",
                "sha256": "b" * 64,
                "mime_type": "image/svg+xml",
                "caption": "Embedding cache lifecycle",
                "role_hint": "flow_diagram",
                "surrounding_text": "Unchanged visual facts reuse cached vectors.",
            },
            "facts": {
                "entities": [{"name": "EmbeddingCache", "type": "component"}],
                "claims": [
                    {
                        "text": "The vector store reuses unchanged embeddings.",
                        "evidence": "cache-hit edge",
                    }
                ],
            },
            "bindings": [
                {
                    "source_path": "codenib/index/embedding/builders.py",
                    "symbol": "EmbeddingsCache",
                    "entity_name": "EmbeddingCache",
                    "evidence": "exact symbol match",
                }
            ],
            "search_text": "embedding vector cache reuse EmbeddingsCache",
        },
    ]
    payload = {
        "schema": "codenib.multimodal-knowledge-view.v1",
        "version": 1,
        "media_manifest_sha256": "c" * 64,
        "visual_facts_manifest_sha256": "d" * 64,
        "grounding_manifest_sha256": "e" * 64,
        "entry_count": len(entries),
        "entries": entries,
    }
    payload["view_sha256"] = _sha256_json(payload)
    return payload


def _rehash_view(view):
    view["view_sha256"] = _sha256_json(
        {key: value for key, value in view.items() if key != "view_sha256"}
    )
    return view


def test_build_and_search_visual_vector_index():
    index = build_visual_vector_index(_knowledge_view(), dimensions=32)

    assert index["entry_count"] == 2
    assert index["embedded_record_count"] == 2
    assert index["reused_record_count"] == 0
    assert all(len(record["embedding"]) == 32 for record in index["records"])

    results = search_visual_vector_index(
        index,
        "embedding vector cache reuse",
        limit=1,
    )

    assert results[0]["artifact_path"] == "docs/embedding-cache.svg"
    assert results[0]["source_paths"] == ["codenib/index/embedding/builders.py"]


def test_visual_vector_index_reuses_only_unchanged_entries():
    initial = build_visual_vector_index(_knowledge_view(), dimensions=32)
    updated_view = copy.deepcopy(_knowledge_view())
    updated_view["entries"][1]["artifact"][
        "caption"
    ] = "Incremental embedding cache lifecycle"
    _rehash_view(updated_view)

    updated = build_visual_vector_index(
        updated_view,
        previous_index=initial,
        dimensions=32,
    )

    assert updated["reused_record_count"] == 1
    assert updated["embedded_record_count"] == 1
    initial_records = {record["artifact_path"]: record for record in initial["records"]}
    updated_records = {record["artifact_path"]: record for record in updated["records"]}
    assert (
        updated_records["docs/architecture.svg"]["embedding_sha256"]
        == initial_records["docs/architecture.svg"]["embedding_sha256"]
    )
    assert (
        updated_records["docs/embedding-cache.svg"]["entry_sha256"]
        != initial_records["docs/embedding-cache.svg"]["entry_sha256"]
    )


def test_visual_vector_index_policy_change_invalidates_reuse():
    initial = build_visual_vector_index(_knowledge_view(), dimensions=16)

    updated = build_visual_vector_index(
        _knowledge_view(),
        previous_index=initial,
        dimensions=32,
    )

    assert updated["reused_record_count"] == 0
    assert updated["embedded_record_count"] == 2

    revised = build_visual_vector_index(
        _knowledge_view(),
        previous_index=updated,
        dimensions=32,
        model_revision="revision-2",
    )
    assert revised["reused_record_count"] == 0
    assert revised["embedded_record_count"] == 2


def test_visual_vector_index_rejects_a_tampered_knowledge_view():
    view = _knowledge_view()
    view["entries"][0]["artifact"]["caption"] = "tampered"

    with pytest.raises(ValueError, match="knowledge view hash"):
        build_visual_vector_index(view)


def test_visual_vector_index_round_trip_and_tamper_detection(tmp_path):
    path = tmp_path / "visual-vector-index.json"
    index = build_visual_vector_index(_knowledge_view(), dimensions=16)

    save_visual_vector_index(index, path)
    loaded = load_visual_vector_index(path)

    assert loaded == index
    tampered = copy.deepcopy(loaded)
    tampered["records"][0]["embedding"][0] = 1.0
    with pytest.raises(ValueError, match="embedding hash"):
        validate_visual_vector_index(tampered)


def test_non_local_visual_vector_policy_uses_the_configured_embedder():
    seen_documents = []

    def embed_documents(documents):
        seen_documents.extend(documents)
        return [
            (
                [1.0, 0.0]
                if "Embedding" in document["text"] or "cache" in document["text"]
                else [0.0, 1.0]
            )
            for document in documents
        ]

    def embed_queries(texts):
        return [[1.0, 0.0] if "cache" in text else [0.0, 1.0] for text in texts]

    index = build_visual_vector_index(
        _knowledge_view(),
        provider="test-provider",
        model="test-embedding-model",
        dimensions=2,
        document_embedder=embed_documents,
        document_modalities=("image", "text"),
    )

    assert index["embedding_policy"]["document_modalities"] == ["image", "text"]
    assert index["embedding_policy"]["query_modality"] == "text"
    assert seen_documents[0]["artifact_path"] == "docs/architecture.svg"
    assert seen_documents[0]["artifact_sha256"] == "a" * 64
    assert seen_documents[0]["mime_type"] == "image/svg+xml"
    assert "Request router architecture" in seen_documents[0]["text"]

    with pytest.raises(ValueError, match="require an embedder"):
        search_visual_vector_index(index, "cache")
    results = search_visual_vector_index(index, "cache", embedder=embed_queries)
    assert results[0]["artifact_path"] == "docs/embedding-cache.svg"


def test_update_multimodal_knowledge_writes_bundle_and_vector_sidecar(
    tmp_path,
    capsys,
):
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "docs" / "architecture.svg").write_text(
        "<svg>WikiServer Router</svg>",
        encoding="utf-8",
    )
    (repo / "README.md").write_text(
        "![WikiServer request flow](docs/architecture.svg)",
        encoding="utf-8",
    )
    (repo / "src" / "wiki.py").write_text(
        "class WikiServer: pass",
        encoding="utf-8",
    )
    bundle_path = tmp_path / "multimodal-bundle.json"
    vector_path = tmp_path / "visual-vector-index.json"

    result = update_multimodal_knowledge(
        [
            str(repo),
            "--bundle-output",
            str(bundle_path),
            "--visual-vector-output",
            str(vector_path),
            "--commit",
            "abc123",
        ]
    )

    assert result == 0
    assert bundle_path.is_file()
    assert load_visual_vector_index(vector_path)["entry_count"] == 1
    counts = json.loads(capsys.readouterr().out)
    assert counts["visual_vector_embedded_records"] == 1

    second_bundle_path = tmp_path / "second-bundle.json"
    second_vector_path = tmp_path / "second-vector-index.json"
    result = update_multimodal_knowledge(
        [
            str(repo),
            "--bundle-output",
            str(second_bundle_path),
            "--visual-vector-output",
            str(second_vector_path),
            "--previous-visual-vector-index",
            str(vector_path),
            "--commit",
            "abc123",
        ]
    )

    assert result == 0
    counts = json.loads(capsys.readouterr().out)
    assert counts["visual_vector_reused_records"] == 1


def test_update_cli_builds_wemm_vector_options(tmp_path):
    args = build_parser().parse_args(
        [
            str(tmp_path),
            "--bundle-output",
            str(tmp_path / "bundle.json"),
            "--visual-vector-output",
            str(tmp_path / "vectors.json"),
            "--visual-vector-backend",
            "wemm",
            "--visual-vector-revision",
            "a" * 40,
            "--visual-vector-trust-remote-code",
            "--visual-vector-dimensions",
            "256",
        ]
    )

    options = _visual_vector_options(args, repo_path=tmp_path)

    assert options["provider"] == "wemm/sentence-transformers"
    assert options["model"] == "tencent/WeMM-Embedding-2B"
    assert options["model_revision"] == "a" * 40
    assert options["document_modalities"] == ("image", "text")
    assert callable(options["document_embedder"])
