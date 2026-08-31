# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from codenib.wiki.media_archify import (
    compile_visual_graph_plan_to_archify,
    save_archify_architecture,
)
from codenib.wiki.media_graph_plan import build_visual_graph_plan


def _plan():
    return build_visual_graph_plan(
        {
            "artifact": {"path": "docs/runtime.svg"},
            "facts": {
                "artifact_path": "docs/runtime.svg",
                "entities": [
                    {"name": "WikiRuntime", "evidence": "visual label"},
                    {"name": "FactIndex", "evidence": "visual label"},
                ],
                "relations": [
                    {
                        "source": "WikiRuntime",
                        "target": "FactIndex",
                        "relation": "queries",
                        "evidence": "authored arrow",
                    }
                ],
            },
            "bindings": [
                {
                    "entity_name": "WikiRuntime",
                    "source_path": "codenib/web/app.py",
                    "symbol": "wiki_page",
                    "line": 100,
                    "score": 0.95,
                    "evidence": "exact definition",
                },
                {
                    "entity_name": "FactIndex",
                    "source_path": "codenib/graph/fact_query.py",
                    "symbol": "FactQueryIndex",
                    "line": 40,
                    "score": 0.4,
                    "evidence": "weak candidate",
                },
            ],
        }
    )


def test_archify_export_preserves_explicit_topology_and_strong_sources():
    document = compile_visual_graph_plan_to_archify(
        _plan(),
        repository_url="https://github.com/sysevol-ai/CodeNib",
        revision="a" * 40,
    )

    assert document["schema_version"] == 1
    assert document["diagram_type"] == "architecture"
    assert document["meta"]["repository"] == {
        "url": "https://github.com/sysevol-ai/CodeNib",
        "revision": "a" * 40,
    }
    components = {item["id"]: item for item in document["components"]}
    assert components["WikiRuntime"]["sources"] == [
        {
            "path": "codenib/web/app.py",
            "line": 100,
            "label": "wiki_page",
        }
    ]
    assert "sources" not in components["FactIndex"]
    assert document["connections"] == [
        {
            "id": "relation-001",
            "from": "WikiRuntime",
            "to": "FactIndex",
            "label": "queries",
            "route": "auto",
        }
    ]


def test_archify_export_without_repository_does_not_claim_source_verification():
    document = compile_visual_graph_plan_to_archify(_plan())

    assert "repository" not in document["meta"]
    assert all("sources" not in component for component in document["components"])


@pytest.mark.parametrize(
    "options,match",
    [
        ({"repository_url": "https://github.com/org/repo"}, "set together"),
        (
            {"repository_url": "https://example.com/org/repo", "revision": "a" * 40},
            "GitHub",
        ),
        (
            {
                "repository_url": "https://github.com/org/repo",
                "revision": "short",
            },
            "40-character",
        ),
        ({"minimum_grounding_score": float("nan")}, "finite"),
    ],
)
def test_archify_export_rejects_unverifiable_repository_metadata(options, match):
    with pytest.raises(ValueError, match=match):
        compile_visual_graph_plan_to_archify(_plan(), **options)


def test_save_archify_architecture_is_canonical_and_preserves_mode(tmp_path):
    output = tmp_path / "runtime.architecture.json"
    output.write_text("old", encoding="utf-8")
    output.chmod(0o640)
    document = compile_visual_graph_plan_to_archify(_plan())

    save_archify_architecture(document, output)

    assert json.loads(output.read_text(encoding="utf-8")) == document
    assert output.stat().st_mode & 0o777 == 0o640
