# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from codenib.index.embedding.artifact_integrity import (
    capture_authenticated_vector_view,
    validate_vector_generation_artifacts,
)
from codenib.native_index_authorization import _mint_trusted_local_admin_authorization
from codenib.wiki import (
    build_visual_vector_index,
    create_visual_vector_store,
    update_visual_vector_store,
    visual_vector_documents,
    visual_vector_search_results,
)
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


def _knowledge_view(count=12):
    entries = []
    for index in range(count):
        entries.append(
            {
                "artifact": {
                    "path": f"docs/diagram-{index}.svg",
                    "sha256": f"{index + 1:064x}",
                    "mime_type": "image/svg+xml",
                    "caption": f"Component {index} request flow",
                    "role_hint": "architecture_diagram",
                    "surrounding_text": f"Component{index} handles route{index}.",
                },
                "facts": {
                    "entities": [{"name": f"Component{index}", "type": "component"}],
                    "claims": [
                        {
                            "text": f"route{index} reaches Component{index}",
                            "evidence": f"arrow {index}",
                        }
                    ],
                },
                "bindings": [
                    {
                        "source_path": f"src/component_{index}.py",
                        "symbol": f"Component{index}",
                        "entity_name": f"Component{index}",
                        "evidence": "exact symbol match",
                    }
                ],
                "search_text": f"route{index} Component{index} request flow",
            }
        )
    payload = {
        "schema": "codenib.multimodal-knowledge-view.v1",
        "version": 1,
        "media_manifest_sha256": "a" * 64,
        "visual_facts_manifest_sha256": "b" * 64,
        "grounding_manifest_sha256": "c" * 64,
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


def _authorization(store, path):
    with capture_authenticated_vector_view(path) as view:
        return _mint_trusted_local_admin_authorization(
            view.ownership,
            view_type="vector",
            semantic_contract=store.artifact_metadata,
            evidence=("visual-vector-store-unit-test",),
        )


def test_materializes_searchable_source_grounded_faiss_store():
    index = build_visual_vector_index(_knowledge_view(), dimensions=64)
    store = create_visual_vector_store(index)

    stats = update_visual_vector_store(store, index)
    results = visual_vector_search_results(store, index, "route7 Component7", limit=3)

    assert stats == {
        "mode": "rebuild",
        "entry_count": 12,
        "changed_entry_count": 12,
        "embedding_policy_sha256": index["embedding_policy_sha256"],
    }
    assert store.l2_index.ntotal == 12
    assert results[0]["artifact_path"] == "docs/diagram-7.svg"
    assert results[0]["source_paths"] == ["src/component_7.py"]
    assert results[0]["symbols"] == ["Component7"]


def test_small_visual_change_uses_code_vector_store_delta_path(tmp_path):
    initial = build_visual_vector_index(_knowledge_view(), dimensions=32)
    store = create_visual_vector_store(initial)
    update_visual_vector_store(store, initial)
    unchanged_vector = store.l2_index.reconstruct(0).copy()

    changed_view = copy.deepcopy(_knowledge_view())
    changed_view["entries"][7]["artifact"][
        "caption"
    ] = "Updated Component 7 request flow"
    _rehash_view(changed_view)
    current = build_visual_vector_index(
        changed_view,
        previous_index=initial,
        dimensions=32,
    )

    stats = update_visual_vector_store(
        store,
        current,
        previous_index=initial,
        threshold=0.1,
    )

    assert stats["mode"] == "delta"
    assert stats["changed_entry_count"] == 1
    assert store.l2_index.ntotal == 12
    assert [document.metadata["chunk_id"] for document in store.l2_documents] == list(
        range(12)
    )
    assert store.l2_index.reconstruct(0) == pytest.approx(unchanged_vector)

    output = (tmp_path / "delta-store").resolve()
    store.save(output)
    reopened = create_visual_vector_store(current, store_path=output)
    reopened.load(native_index_authorization=_authorization(reopened, output))
    changed_text = next(
        record["embedding_text"]
        for record in current["records"]
        if record["artifact_path"] == "docs/diagram-7.svg"
    )
    results = visual_vector_search_results(
        reopened,
        current,
        changed_text,
        limit=1,
    )
    assert results[0]["artifact_path"] == "docs/diagram-7.svg"


def test_schema_8_store_round_trip_uses_inert_json(tmp_path):
    index = build_visual_vector_index(_knowledge_view(2), dimensions=16)
    output = (tmp_path / "visual-faiss").resolve()
    store = create_visual_vector_store(index, store_path=output)
    update_visual_vector_store(store, index)
    store.save()

    model_suffix = store.embedding_model.replace("/", "__")
    assert (output / "l2" / f"documents_{model_suffix}.json").is_file()
    assert not (output / "l2" / f"documents_{model_suffix}.pkl").exists()
    validate_vector_generation_artifacts(output, model_suffix)

    reopened = create_visual_vector_store(index, store_path=output)
    reopened.load(native_index_authorization=_authorization(reopened, output))
    results = visual_vector_search_results(
        reopened,
        index,
        "route1 Component1",
        limit=1,
    )
    assert results[0]["artifact_path"] == "docs/diagram-1.svg"


def test_non_local_store_requires_query_embedder():
    def documents(values):
        return [[1.0, 0.0] for _ in values]

    index = build_visual_vector_index(
        _knowledge_view(1),
        provider="wemm",
        model="Tencent/WeMM-Embedding-2B",
        dimensions=2,
        document_embedder=documents,
        document_modalities=("image", "text"),
    )

    with pytest.raises(ValueError, match="require a query embedder"):
        create_visual_vector_store(index)

    store = create_visual_vector_store(
        index,
        query_embedder=lambda texts: [[1.0, 0.0] for _ in texts],
    )
    update_visual_vector_store(store, index)
    assert visual_vector_search_results(store, index, "component", limit=1)


def test_documents_have_the_strict_schema_8_row_shape():
    index = build_visual_vector_index(_knowledge_view(1), dimensions=8)
    documents, vectors = visual_vector_documents(index)

    assert len(vectors[0]) == 8
    assert set(documents[0].metadata) == {
        "chunk_id",
        "chunk_type",
        "name",
        "file",
        "start_line",
        "end_line",
        "node_id",
        "level",
        "content_hash",
    }
    assert documents[0].metadata["file"] == "docs/diagram-0.svg"


def test_search_rejects_unbounded_queries_and_limits():
    index = build_visual_vector_index(_knowledge_view(1), dimensions=8)
    store = create_visual_vector_store(index)
    update_visual_vector_store(store, index)

    assert visual_vector_search_results(store, index, "   ") == []
    with pytest.raises(ValueError, match="byte limit"):
        visual_vector_search_results(store, index, "x" * 8193)
    with pytest.raises(ValueError, match="between 1 and 50"):
        visual_vector_search_results(store, index, "component", limit=51)


def test_rejects_a_store_bound_to_another_policy():
    small = build_visual_vector_index(_knowledge_view(1), dimensions=8)
    large = build_visual_vector_index(_knowledge_view(1), dimensions=16)
    store = create_visual_vector_store(small)

    with pytest.raises(ValueError, match="identity does not match"):
        update_visual_vector_store(store, large)


def test_incremental_cli_materializes_and_reopens_faiss_store(tmp_path, capsys):
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
    bundle = tmp_path / "bundle.json"
    sidecar = tmp_path / "visual-vectors.json"
    store = (tmp_path / "visual-faiss").resolve()
    common = [
        str(repo),
        "--bundle-output",
        str(bundle),
        "--visual-vector-output",
        str(sidecar),
        "--visual-vector-store-output",
        str(store),
    ]

    assert update_multimodal_knowledge(common) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["visual_vector_store_records"] == 1
    assert first["visual_vector_store_update_mode"] == "rebuild"

    assert (
        update_multimodal_knowledge(
            [
                *common,
                "--previous-visual-vector-index",
                str(sidecar),
                "--previous-visual-vector-store",
                str(store),
            ]
        )
        == 0
    )
    second = json.loads(capsys.readouterr().out)
    assert second["visual_vector_reused_records"] == 1
    assert second["visual_vector_store_changed_records"] == 0
    assert second["visual_vector_store_update_mode"] == "delta"
    assert list((store / "l2").glob("index_*.faiss"))
