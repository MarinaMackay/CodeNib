# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from codenib.wiki.media_graph_plan import build_visual_graph_plan
from codenib.wiki.media_storyboard import (
    build_visual_storyboard,
    build_visual_storyboard_manifest,
    compile_visual_storyboard_to_markdown,
    validate_visual_storyboard,
)


def _plan():
    return build_visual_graph_plan(
        {
            "artifact": {
                "path": "docs/architecture.svg",
                "embedded_text": "WikiRenderer calls IndexCompiler",
            },
            "facts": {
                "entities": [
                    {"name": "WikiRenderer", "type": "component"},
                    {"name": "IndexCompiler", "type": "component"},
                ],
                "relations": [],
            },
            "bindings": [
                {
                    "entity_name": "WikiRenderer",
                    "source_path": "src/wiki.py",
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
            ],
        }
    )


def test_build_visual_storyboard_from_graph_plan():
    storyboard = build_visual_storyboard(_plan())

    assert storyboard["schema"] == "codenib.visual-storyboard.v1"
    assert storyboard["artifact_path"] == "docs/architecture.svg"
    assert len(storyboard["frames"]) >= 3
    assert storyboard["frames"][0]["id"] == "orient"
    assert storyboard["frames"][-1]["id"] == "grounding"
    assert "src/wiki.py" in storyboard["frames"][-1]["source_citations"]
    assert (
        validate_visual_storyboard(storyboard)["storyboard_sha256"]
        == storyboard["storyboard_sha256"]
    )


def test_build_visual_storyboard_manifest_wraps_storyboards():
    manifest = build_visual_storyboard_manifest(
        {
            "manifest_sha256": "graph-hash",
            "plans": [_plan()],
        }
    )

    assert manifest["schema"] == "codenib.visual-storyboard-manifest.v1"
    assert manifest["visual_graph_manifest_sha256"] == "graph-hash"
    assert manifest["storyboard_count"] == 1
    assert manifest["manifest_sha256"]


def test_compile_visual_storyboard_to_markdown():
    markdown = compile_visual_storyboard_to_markdown(build_visual_storyboard(_plan()))

    assert "WikiRenderer" in markdown
    assert "src/wiki.py" in markdown
    assert "ms" in markdown


def test_validate_visual_storyboard_rejects_bad_paths_and_control_text():
    storyboard = build_visual_storyboard(_plan())
    storyboard["frames"][0]["source_citations"] = ["../secret.py"]

    with pytest.raises(ValueError, match="repository-relative"):
        validate_visual_storyboard(storyboard)

    storyboard = build_visual_storyboard(_plan())
    storyboard["frames"][0]["title"] = "bad\x00title"

    with pytest.raises(ValueError, match="control"):
        validate_visual_storyboard(storyboard)
