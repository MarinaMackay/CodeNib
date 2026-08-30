# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

import codenib.wiki.media_graph_plan as media_graph_plan
from codenib.wiki.media_graph_plan import (
    build_visual_graph_manifest,
    build_visual_graph_plan,
    compile_visual_graph_plan_to_mermaid,
    load_visual_graph_manifest,
    save_visual_graph_manifest,
    validate_visual_graph_manifest,
    validate_visual_graph_plan,
)
from codenib.wiki.media_knowledge import build_multimodal_knowledge_view


def _entry():
    return {
        "artifact": {
            "path": "docs/architecture.svg",
            "caption": "WikiRenderer architecture",
        },
        "facts": {
            "artifact_path": "docs/architecture.svg",
            "entities": [
                {
                    "name": "WikiRenderer",
                    "type": "component",
                    "evidence": "visual label",
                },
                {
                    "name": "IndexCompiler",
                    "type": "component",
                    "evidence": "visual label",
                },
                {
                    "name": "VectorStore",
                    "type": "data_store",
                    "evidence": "visual label",
                },
            ],
            "relations": [
                {
                    "source": "WikiRenderer",
                    "target": "IndexCompiler",
                    "relation": "calls",
                    "evidence": "arrow from renderer to compiler",
                },
                {
                    "source": "IndexCompiler",
                    "target": "VectorStore",
                    "relation": "writes_to",
                    "evidence": "labeled arrow",
                },
            ],
        },
        "bindings": [
            {
                "entity_name": "WikiRenderer",
                "source_path": "src/wiki.py",
                "symbol": "WikiRenderer",
                "line": 8,
                "score": 0.7,
                "evidence": "partial symbol match",
            },
            {
                "entity_name": "WikiRenderer",
                "source_path": "src/runtime.py",
                "symbol": "WikiRenderer",
                "line": 4,
                "score": 1.0,
                "evidence": "exact symbol match",
            },
            {
                "entity_name": "IndexCompiler",
                "source_path": "src/compiler.py",
                "symbol": "IndexCompiler",
                "line": 15,
                "score": 1.0,
                "evidence": "exact symbol match",
            },
            {
                "entity_name": "VectorStore",
                "source_path": "src/vector.py",
                "symbol": "VectorStore",
                "line": 21,
                "score": 1.0,
                "evidence": "exact symbol match",
            },
        ],
    }


def _knowledge_view(*entries):
    artifacts = [entry["artifact"] for entry in entries]
    facts = [entry["facts"] for entry in entries]
    bindings = []
    for entry in entries:
        bindings.extend(
            {"artifact_path": entry["artifact"]["path"], **binding}
            for binding in entry["bindings"]
        )
    return build_multimodal_knowledge_view(
        {"manifest_sha256": "a" * 64, "artifacts": artifacts},
        {"manifest_sha256": "b" * 64, "facts": facts},
        {"manifest_sha256": "c" * 64, "bindings": bindings},
    )


def test_build_visual_graph_plan_uses_explicit_facts_and_best_grounding():
    plan = build_visual_graph_plan(_entry())

    assert plan["schema"] == "codenib.visual-graph-plan.v1"
    assert len(plan["plan_sha256"]) == 64
    nodes = {node["label"]: node for node in plan["nodes"]}
    assert nodes["WikiRenderer"]["source_path"] == "src/runtime.py"
    assert nodes["WikiRenderer"]["line"] == 4
    assert nodes["WikiRenderer"]["grounding_score"] == 1.0
    assert nodes["WikiRenderer"]["grounding_evidence"] == "exact symbol match"
    assert {
        (edge["source"], edge["target"], edge["relation"]) for edge in plan["edges"]
    } == {
        ("WikiRenderer", "IndexCompiler", "calls"),
        ("IndexCompiler", "VectorStore", "writes_to"),
    }
    assert validate_visual_graph_plan(plan) == plan


def test_graph_plan_does_not_invent_edges_from_caption_text():
    entry = _entry()
    entry["facts"]["relations"] = []
    entry["artifact"]["caption"] = "WikiRenderer calls IndexCompiler"

    plan = build_visual_graph_plan(entry)

    assert plan["edges"] == []


def test_node_ids_are_unique_after_mermaid_normalization():
    entry = _entry()
    entry["facts"]["entities"] = [
        {"name": "A-B", "evidence": "left"},
        {"name": "A B", "evidence": "right"},
    ]
    entry["facts"]["relations"] = []
    entry["bindings"] = []

    plan = build_visual_graph_plan(entry)

    ids = [node["id"] for node in plan["nodes"]]
    assert len(ids) == len(set(ids)) == 2
    assert ids[0] == "A_B"
    assert ids[1].startswith("A_B_")


def test_compile_mermaid_escapes_untrusted_labels_and_relations():
    entry = _entry()
    entry["facts"]["entities"][0]["name"] = 'Wiki"] --> injected["node'
    entry["facts"]["relations"] = [
        {
            "source": 'Wiki"] --> injected["node',
            "target": "IndexCompiler",
            "relation": "calls|breakout",
            "evidence": "arrow",
        }
    ]
    entry["bindings"][0]["entity_name"] = 'Wiki"] --> injected["node'
    entry["bindings"][1]["entity_name"] = 'Wiki"] --> injected["node'

    mermaid = compile_visual_graph_plan_to_mermaid(build_visual_graph_plan(entry))

    assert "&quot;" in mermaid
    assert "&#91;" in mermaid
    assert "&#93;" in mermaid
    assert "calls&#124;breakout" in mermaid
    assert 'Wiki"] --> injected' not in mermaid


def test_build_and_round_trip_visual_graph_manifest(tmp_path):
    manifest = build_visual_graph_manifest(_knowledge_view(_entry()))
    output = tmp_path / "graphs.json"

    save_visual_graph_manifest(manifest, output)

    assert load_visual_graph_manifest(output) == manifest
    assert manifest["plan_count"] == 1
    assert len(manifest["knowledge_view_sha256"]) == 64
    assert len(manifest["manifest_sha256"]) == 64


def test_save_visual_graph_manifest_preserves_existing_mode(tmp_path):
    manifest = build_visual_graph_manifest(_knowledge_view(_entry()))
    output = tmp_path / "graphs.json"
    output.write_text("old", encoding="utf-8")
    output.chmod(0o640)

    save_visual_graph_manifest(manifest, output)

    assert output.stat().st_mode & 0o777 == 0o640


def test_visual_graph_plan_rejects_tampering_extra_fields_and_bad_paths():
    plan = build_visual_graph_plan(_entry())
    plan["nodes"][0]["label"] = "tampered"
    with pytest.raises(ValueError, match="hash does not match"):
        validate_visual_graph_plan(plan)

    plan = build_visual_graph_plan(_entry())
    plan["unexpected"] = True
    with pytest.raises(ValueError, match="fields are invalid"):
        validate_visual_graph_plan(plan)

    entry = _entry()
    entry["bindings"][0]["source_path"] = "../secret.py"
    with pytest.raises(ValueError, match="repository-relative"):
        build_visual_graph_plan(entry)

    plan = build_visual_graph_plan(_entry())
    plan["nodes"][0]["grounding_score"] = float("nan")
    with pytest.raises(ValueError, match="grounding_score is invalid"):
        validate_visual_graph_plan(plan)


def test_visual_graph_plan_rejects_dangling_and_duplicate_edges():
    plan = build_visual_graph_plan(_entry())
    plan["edges"][0]["target"] = "Missing"
    plan["plan_sha256"] = media_graph_plan._sha256_json(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    with pytest.raises(ValueError, match="endpoints"):
        validate_visual_graph_plan(plan)

    plan = build_visual_graph_plan(_entry())
    plan["edges"].append(dict(plan["edges"][0]))
    plan["plan_sha256"] = media_graph_plan._sha256_json(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    with pytest.raises(ValueError, match="unique"):
        validate_visual_graph_plan(plan)


def test_visual_graph_manifest_rejects_wrong_view_hash_and_duplicate_json(tmp_path):
    view = _knowledge_view(_entry())
    view["view_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="view hash does not match"):
        build_visual_graph_manifest(view)

    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_visual_graph_manifest(path)


def test_visual_graph_manifest_rejects_count_and_hash_tampering():
    manifest = build_visual_graph_manifest(_knowledge_view(_entry()))
    manifest["plan_count"] = 2
    with pytest.raises(ValueError, match="plan_count"):
        validate_visual_graph_manifest(manifest)

    manifest = build_visual_graph_manifest(_knowledge_view(_entry()))
    manifest["plans"][0]["nodes"][0]["label"] = "tampered"
    with pytest.raises(ValueError, match="plan hash does not match"):
        validate_visual_graph_manifest(manifest)


def test_visual_graph_plan_rejects_generators_and_bounds_nodes(monkeypatch):
    entry = _entry()
    entry["facts"]["entities"] = (
        {"name": f"Node{index}", "evidence": "generated"} for index in range(3)
    )
    with pytest.raises(ValueError, match="bounded array"):
        build_visual_graph_plan(entry)

    monkeypatch.setattr(media_graph_plan, "_MAX_NODES_PER_PLAN", 1)
    entry = _entry()
    plan = build_visual_graph_plan(entry)
    assert len(plan["nodes"]) == 1


def test_visual_graph_manifest_is_canonical_json():
    manifest = build_visual_graph_manifest(_knowledge_view(_entry()))

    assert json.loads(json.dumps(manifest, allow_nan=False)) == manifest
