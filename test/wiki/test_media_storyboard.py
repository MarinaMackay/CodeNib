# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

import codenib.wiki.media_storyboard as media_storyboard
from codenib.wiki.media_graph_plan import build_visual_graph_plan
from codenib.wiki.media_storyboard import (
    build_visual_storyboard,
    build_visual_storyboard_manifest,
    compile_visual_storyboard_to_markdown,
    load_visual_storyboard_manifest,
    save_visual_storyboard_manifest,
    validate_visual_storyboard,
    validate_visual_storyboard_manifest,
)


def _entry(*, relations=True):
    return {
        "artifact": {"path": "docs/request-flow.svg"},
        "facts": {
            "artifact_path": "docs/request-flow.svg",
            "entities": [
                {"name": "Router", "evidence": "visual label"},
                {"name": "WikiBuilder", "evidence": "visual label"},
            ],
            "relations": (
                [
                    {
                        "source": "Router",
                        "target": "WikiBuilder",
                        "relation": "calls",
                        "evidence": "arrow",
                    }
                ]
                if relations
                else []
            ),
        },
        "bindings": [
            {
                "entity_name": "Router",
                "source_path": "codenib/web/app.py",
                "symbol": "wiki_page",
                "line": 120,
                "score": 1.0,
                "evidence": "exact symbol match",
            },
            {
                "entity_name": "WikiBuilder",
                "source_path": "codenib/wiki/builder.py",
                "symbol": "WikiBuilder",
                "line": 30,
                "score": 1.0,
                "evidence": "exact symbol match",
            },
        ],
    }


def _plan(*, relations=True):
    return build_visual_graph_plan(_entry(relations=relations))


def _graph_manifest(*plans):
    payload = {
        "schema": "codenib.visual-graph-manifest.v1",
        "version": 1,
        "knowledge_view_sha256": "a" * 64,
        "plan_count": len(plans),
        "plans": sorted(plans, key=lambda plan: plan["artifact_path"]),
    }
    payload["manifest_sha256"] = media_storyboard._sha256_json(payload)
    return payload


def _rehash(storyboard):
    storyboard["storyboard_sha256"] = media_storyboard._sha256_json(
        {key: value for key, value in storyboard.items() if key != "storyboard_sha256"}
    )


def test_storyboard_uses_explicit_relation_and_source_citations():
    plan = _plan()

    storyboard = build_visual_storyboard(plan)

    assert storyboard["graph_plan_sha256"] == plan["plan_sha256"]
    assert [frame["kind"] for frame in storyboard["frames"]] == [
        "overview",
        "relation",
        "source_ledger",
    ]
    relation = storyboard["frames"][1]
    assert relation["focus_node_ids"] == ["Router", "WikiBuilder"]
    assert relation["source_citations"] == [
        {
            "source_path": "codenib/web/app.py",
            "symbol": "wiki_page",
            "line": 120,
        },
        {
            "source_path": "codenib/wiki/builder.py",
            "symbol": "WikiBuilder",
            "line": 30,
        },
    ]
    assert validate_visual_storyboard(storyboard) == storyboard


def test_storyboard_without_relations_uses_entity_frames_without_inventing_edges():
    storyboard = build_visual_storyboard(_plan(relations=False))

    assert [frame["kind"] for frame in storyboard["frames"]] == [
        "overview",
        "entity",
        "entity",
        "source_ledger",
    ]
    assert all("related" not in frame["narration"] for frame in storyboard["frames"])


def test_storyboard_keeps_ungrounded_entities_explicit():
    entry = _entry(relations=False)
    entry["bindings"] = []

    storyboard = build_visual_storyboard(build_visual_graph_plan(entry))

    entity = storyboard["frames"][1]
    assert entity["source_citations"] == []
    assert "not yet sufficiently source-grounded" in entity["visual_prompt"]
    assert "pending" in storyboard["frames"][-1]["narration"]


def test_storyboard_does_not_promote_weak_lexical_binding_to_citation():
    entry = _entry(relations=False)
    entry["bindings"][0]["score"] = 0.75
    entry["bindings"][0]["evidence"] = "partial symbol match"

    storyboard = build_visual_storyboard(build_visual_graph_plan(entry))

    router = storyboard["frames"][1]
    assert router["title"] == "Router"
    assert router["source_citations"] == []
    assert "not yet sufficiently source-grounded" in router["visual_prompt"]
    assert "pending" not in storyboard["frames"][-1]["narration"]


def test_storyboard_manifest_round_trip_and_mode_preservation(tmp_path):
    manifest = build_visual_storyboard_manifest(_graph_manifest(_plan()))
    output = tmp_path / "storyboards.json"
    output.write_text("old", encoding="utf-8")
    output.chmod(0o640)

    save_visual_storyboard_manifest(manifest, output)

    assert output.stat().st_mode & 0o777 == 0o640
    assert load_visual_storyboard_manifest(output) == manifest
    assert manifest["storyboard_count"] == 1


def test_markdown_shot_list_contains_prompts_and_precise_sources():
    markdown = compile_visual_storyboard_to_markdown(build_visual_storyboard(_plan()))

    assert "Kind: `relation`" in markdown
    assert "Visual direction:" in markdown
    assert "`codenib/web/app.py:120` (`wiki_page`)" in markdown


def test_markdown_shot_list_escapes_untrusted_vlm_text():
    storyboard = build_visual_storyboard(_plan())
    storyboard["frames"][0]["title"] = '<script>alert("x")</script> [link](bad)'
    _rehash(storyboard)

    markdown = compile_visual_storyboard_to_markdown(storyboard)

    assert "<script>" not in markdown
    assert "&lt;script&gt;" in markdown
    assert r"\[link\]\(bad\)" in markdown


def test_storyboard_rejects_unknown_nodes_and_citations():
    plan = _plan()
    storyboard = build_visual_storyboard(plan)
    allowed_citations = {
        (node["source_path"], node["symbol"], node["line"])
        for node in plan["nodes"]
        if node["source_path"]
    }

    storyboard["frames"][0]["focus_node_ids"] = ["Missing"]
    _rehash(storyboard)
    with pytest.raises(ValueError, match="unknown graph node"):
        validate_visual_storyboard(
            storyboard,
            allowed_node_ids={node["id"] for node in plan["nodes"]},
            allowed_citations=allowed_citations,
        )

    storyboard = build_visual_storyboard(plan)
    storyboard["frames"][0]["source_citations"][0]["source_path"] = "fake.py"
    _rehash(storyboard)
    with pytest.raises(ValueError, match="citation is not allowed"):
        validate_visual_storyboard(
            storyboard,
            allowed_node_ids={node["id"] for node in plan["nodes"]},
            allowed_citations=allowed_citations,
        )


def test_storyboard_rejects_tampering_extra_fields_and_unsafe_paths():
    storyboard = build_visual_storyboard(_plan())
    storyboard["frames"][0]["title"] = "tampered"
    with pytest.raises(ValueError, match="hash does not match"):
        validate_visual_storyboard(storyboard)

    storyboard = build_visual_storyboard(_plan())
    storyboard["unexpected"] = True
    with pytest.raises(ValueError, match="fields are invalid"):
        validate_visual_storyboard(storyboard)

    storyboard = build_visual_storyboard(_plan())
    storyboard["artifact_path"] = "../secret.svg"
    _rehash(storyboard)
    with pytest.raises(ValueError, match="repository-relative"):
        validate_visual_storyboard(storyboard)


def test_storyboard_rejects_duplicate_frames_and_invalid_duration():
    storyboard = build_visual_storyboard(_plan())
    storyboard["frames"][1]["id"] = storyboard["frames"][0]["id"]
    _rehash(storyboard)
    with pytest.raises(ValueError, match="frame ids must be unique"):
        validate_visual_storyboard(storyboard)

    storyboard = build_visual_storyboard(_plan())
    storyboard["frames"][0]["duration_ms"] = 0
    storyboard["total_duration_ms"] -= 4000
    _rehash(storyboard)
    with pytest.raises(ValueError, match="duration_ms is invalid"):
        validate_visual_storyboard(storyboard)


def test_storyboard_manifest_rejects_count_hash_and_duplicate_json(tmp_path):
    manifest = build_visual_storyboard_manifest(_graph_manifest(_plan()))
    manifest["storyboard_count"] = 2
    with pytest.raises(ValueError, match="storyboard_count"):
        validate_visual_storyboard_manifest(manifest)

    manifest = build_visual_storyboard_manifest(_graph_manifest(_plan()))
    manifest["storyboards"][0]["title"] = "tampered"
    with pytest.raises(ValueError, match="storyboard hash does not match"):
        validate_visual_storyboard_manifest(manifest)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_visual_storyboard_manifest(duplicate)


def test_storyboard_manifest_checks_graph_hash_and_complete_coverage():
    graph_manifest = _graph_manifest(_plan())
    manifest = build_visual_storyboard_manifest(graph_manifest)

    manifest["storyboards"][0]["graph_plan_sha256"] = "0" * 64
    _rehash(manifest["storyboards"][0])
    manifest["manifest_sha256"] = media_storyboard._sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="graph plan hash does not match"):
        validate_visual_storyboard_manifest(
            manifest, visual_graph_manifest=graph_manifest
        )

    manifest = build_visual_storyboard_manifest(graph_manifest)
    manifest["storyboards"] = []
    manifest["storyboard_count"] = 0
    manifest["manifest_sha256"] = media_storyboard._sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="does not cover every graph plan"):
        validate_visual_storyboard_manifest(
            manifest, visual_graph_manifest=graph_manifest
        )


def test_storyboard_limits_frame_count_and_serializes_as_json():
    plan = _plan()
    relation = dict(plan["edges"][0])
    plan["edges"] = [{**relation, "relation": f"calls_{index}"} for index in range(48)]
    plan["plan_sha256"] = media_storyboard._sha256_json(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )

    storyboard = build_visual_storyboard(plan)

    assert len(storyboard["frames"]) == 12
    assert json.loads(json.dumps(storyboard, allow_nan=False)) == storyboard
