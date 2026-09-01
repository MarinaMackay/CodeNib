# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil

import pytest

from codenib.wiki.media_graph_plan import build_visual_graph_plan
from codenib.wiki.media_storyboard import (
    build_visual_storyboard,
    build_visual_storyboard_manifest,
)
from codenib.wiki.media_video import (
    load_storyboard_video_manifest,
    read_storyboard_video_asset,
    render_visual_storyboard_manifest_videos,
    render_visual_storyboard_video,
)


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
                    "score": 0.92,
                    "evidence": "exact definition",
                },
            ],
        }
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_renders_real_mp4_and_authenticates_manifest(tmp_path):
    storyboard = build_visual_storyboard(_plan())
    output = tmp_path / "runtime.mp4"

    provenance = render_visual_storyboard_video(
        storyboard, output, width=320, height=180, fps=4
    )

    assert output.read_bytes()[4:8] == b"ftyp"
    assert provenance["mime_type"] == "video/mp4"
    assert provenance["frame_count"] == len(storyboard["frames"])
    assert provenance["source_citations"] == [
        "codenib/graph/fact_query.py",
        "codenib/web/app.py",
    ]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_manifest_render_round_trip_detects_video_tampering(tmp_path):
    plan = _plan()
    import codenib.wiki.media_storyboard as media_storyboard

    graph_manifest = {
        "schema": "codenib.visual-graph-manifest.v1",
        "version": 1,
        "knowledge_view_sha256": "a" * 64,
        "plan_count": 1,
        "plans": [plan],
    }
    graph_manifest["manifest_sha256"] = media_storyboard._sha256_json(graph_manifest)
    storyboard_manifest = build_visual_storyboard_manifest(graph_manifest)
    rendered = render_visual_storyboard_manifest_videos(
        storyboard_manifest, tmp_path / "videos", width=320, height=180, fps=4
    )
    manifest_path = tmp_path / "videos" / "manifest.json"

    assert load_storyboard_video_manifest(manifest_path) == rendered
    filename = rendered["videos"][0]["output_path"]
    assert read_storyboard_video_asset(manifest_path, filename)

    video_path = manifest_path.parent / filename
    video_path.write_bytes(video_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="size does not match"):
        load_storyboard_video_manifest(manifest_path)


def test_video_manifest_loader_rejects_duplicate_json_and_links(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema":"a","schema":"b","storyboard_manifest_sha256":"x",'
        '"video_count":0,"videos":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        load_storyboard_video_manifest(manifest)

    target = tmp_path / "target.json"
    target.write_text(json.dumps({}), encoding="utf-8")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ValueError, match="bounded regular file"):
        load_storyboard_video_manifest(link)
