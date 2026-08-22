# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from codenib.wiki.media_graph_plan import (
    build_visual_graph_manifest,
    build_visual_graph_plan,
    compile_visual_graph_plan_to_mermaid,
    validate_visual_graph_plan,
)


def _entry():
    return {
        "artifact": {
            "path": "docs/architecture.svg",
            "caption": "WikiRenderer architecture",
            "embedded_text": "WikiRenderer calls IndexCompiler and VectorStore",
        },
        "facts": {
            "entities": [
                {"name": "WikiRenderer", "type": "component"},
                {"name": "IndexCompiler", "type": "component"},
                {"name": "VectorStore", "type": "component"},
            ],
            "relations": [],
            "claims": [],
        },
        "bindings": [
            {
                "entity_name": "WikiRenderer",
                "source_path": "src/compiler.py",
                "symbol": "WikiRenderer",
                "line": 1,
                "evidence": "exact symbol match",
            },
            {
                "entity_name": "IndexCompiler",
                "source_path": "src/compiler.py",
                "symbol": "IndexCompiler",
                "line": 5,
                "evidence": "exact symbol match",
            },
            {
                "entity_name": "VectorStore",
                "source_path": "src/compiler.py",
                "symbol": "VectorStore",
                "line": 9,
                "evidence": "exact symbol match",
            },
        ],
    }


def test_build_visual_graph_plan_infers_validated_call_edges():
    plan = build_visual_graph_plan(_entry())

    assert plan["schema"] == "codenib.visual-graph-plan.v1"
    assert {node["label"] for node in plan["nodes"]} == {
        "WikiRenderer",
        "IndexCompiler",
        "VectorStore",
    }
    assert {
        (edge["source"], edge["target"], edge["relation"]) for edge in plan["edges"]
    } == {
        ("WikiRenderer", "IndexCompiler", "calls"),
        ("WikiRenderer", "VectorStore", "calls"),
    }
    assert validate_visual_graph_plan(plan)["plan_sha256"] == plan["plan_sha256"]


def test_build_visual_graph_manifest_wraps_plans():
    manifest = build_visual_graph_manifest(
        {
            "view_sha256": "view-hash",
            "entries": [_entry()],
        }
    )

    assert manifest["schema"] == "codenib.visual-graph-manifest.v1"
    assert manifest["knowledge_view_sha256"] == "view-hash"
    assert manifest["plan_count"] == 1
    assert manifest["manifest_sha256"]


def test_compile_visual_graph_plan_to_mermaid():
    mermaid = compile_visual_graph_plan_to_mermaid(build_visual_graph_plan(_entry()))

    assert mermaid.startswith("flowchart LR")
    assert 'WikiRenderer["WikiRenderer"]' in mermaid
    assert "WikiRenderer -->|calls| IndexCompiler" in mermaid


def test_validate_visual_graph_plan_rejects_bad_edges_and_paths():
    plan = build_visual_graph_plan(_entry())
    plan["edges"] = [{"source": "WikiRenderer", "target": "Missing"}]

    with pytest.raises(ValueError, match="endpoints"):
        validate_visual_graph_plan(plan)

    plan = build_visual_graph_plan(_entry())
    plan["nodes"][0]["source_path"] = "../secret.py"

    with pytest.raises(ValueError, match="repository-relative"):
        validate_visual_graph_plan(plan)
