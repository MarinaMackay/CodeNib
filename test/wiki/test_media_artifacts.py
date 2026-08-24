# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib

import pytest

import codenib.wiki.media_artifacts as media_artifacts
from codenib.repository_source_selection import RepositorySourceSelection
from codenib.wiki.media_artifacts import discover_media_manifest


def test_discover_media_manifest_collects_markdown_context(tmp_path):
    repo = tmp_path
    docs = repo / "docs"
    assets = docs / "assets"
    assets.mkdir(parents=True)
    image = assets / "architecture.svg"
    image.write_text("<svg>architecture</svg>", encoding="utf-8")
    (repo / "README.md").write_text(
        "\n".join(
            [
                "# Demo",
                "",
                "The service architecture is shown below.",
                "![Architecture overview](docs/assets/architecture.svg)",
                "It maps API requests to the wiki runtime.",
            ]
        ),
        encoding="utf-8",
    )

    manifest = discover_media_manifest(repo, commit="abc123")

    assert manifest["schema"] == "codenib.media-manifest.v1"
    assert manifest["commit"] == "abc123"
    assert manifest["artifact_count"] == 1
    artifact = manifest["artifacts"][0]
    assert artifact["path"] == "docs/assets/architecture.svg"
    assert artifact["media_type"] == "svg"
    assert artifact["mime_type"] == "image/svg+xml"
    assert artifact["sha256"] == hashlib.sha256(b"<svg>architecture</svg>").hexdigest()
    assert artifact["role_hint"] == "architecture_diagram"
    assert artifact["caption"] == "Architecture overview"
    assert "service architecture" in artifact["surrounding_text"]
    assert artifact["references"] == [
        {
            "markdown_path": "README.md",
            "line": 4,
            "alt_text": "Architecture overview",
            "title": "",
            "surrounding_text": (
                "The service architecture is shown below.\n"
                "![Architecture overview](docs/assets/architecture.svg)\n"
                "It maps API requests to the wiki runtime."
            ),
        }
    ]
    assert manifest["manifest_sha256"]


def test_discover_media_manifest_respects_source_selection(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "screenshot.png").write_bytes(b"png")
    (tmp_path / "hidden").mkdir()
    (tmp_path / "hidden" / "architecture.png").write_bytes(b"hidden")

    manifest = discover_media_manifest(
        tmp_path,
        commit="abc123",
        selection=RepositorySourceSelection(["hidden"]),
    )

    assert [artifact["path"] for artifact in manifest["artifacts"]] == [
        "docs/screenshot.png"
    ]


def test_discover_media_manifest_ignores_external_and_escaping_markdown_links(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "local.webp").write_bytes(b"webp")
    (tmp_path / "docs" / "README.md").write_text(
        "\n".join(
            [
                "![Remote](https://example.com/asset.png)",
                "![Escape](../../secret.png)",
                '<img src="local.webp" alt="Dashboard screenshot">',
            ]
        ),
        encoding="utf-8",
    )

    manifest = discover_media_manifest(tmp_path, commit="abc123")

    assert manifest["artifact_count"] == 1
    artifact = manifest["artifacts"][0]
    assert artifact["path"] == "docs/local.webp"
    assert artifact["role_hint"] == "ui_screenshot"
    assert artifact["caption"] == "Dashboard screenshot"


def test_discover_media_manifest_resolves_safe_parent_markdown_links(tmp_path):
    (tmp_path / "docs" / "guide").mkdir(parents=True)
    (tmp_path / "docs" / "assets").mkdir()
    (tmp_path / "docs" / "assets" / "architecture.svg").write_text(
        "<svg/>", encoding="utf-8"
    )
    (tmp_path / "docs" / "guide" / "README.md").write_text(
        "![Architecture](../assets/architecture.svg)",
        encoding="utf-8",
    )

    manifest = discover_media_manifest(tmp_path, commit="abc123")

    artifact = manifest["artifacts"][0]
    assert artifact["caption"] == "Architecture"
    assert artifact["references"][0]["markdown_path"] == "docs/guide/README.md"


def test_discover_media_manifest_skips_symlinks_and_large_media(tmp_path):
    small = tmp_path / "diagram.png"
    small.write_bytes(b"png")
    linked = tmp_path / "linked.png"
    linked.symlink_to(small)

    manifest = discover_media_manifest(tmp_path, commit="abc123")

    assert [artifact["path"] for artifact in manifest["artifacts"]] == ["diagram.png"]


def test_discover_media_manifest_skips_files_that_exceed_hash_limit(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(media_artifacts, "_MAX_MEDIA_BYTES", 2)
    (tmp_path / "diagram.png").write_bytes(b"png")

    manifest = discover_media_manifest(tmp_path, commit="abc123")

    assert manifest["artifacts"] == []


def test_discover_media_manifest_reuses_generator_exclusions_for_both_scans(tmp_path):
    (tmp_path / "visible").mkdir()
    (tmp_path / "visible" / "diagram.png").write_bytes(b"visible")
    (tmp_path / "excluded").mkdir()
    (tmp_path / "excluded" / "secret.png").write_bytes(b"secret")

    manifest = discover_media_manifest(
        tmp_path,
        commit="abc123",
        exclude_roots=(path for path in [tmp_path / "excluded"]),
    )

    assert [artifact["path"] for artifact in manifest["artifacts"]] == [
        "visible/diagram.png"
    ]


def test_discover_media_manifest_rejects_unbounded_artifact_limits(tmp_path):
    with pytest.raises(ValueError, match="max_artifacts"):
        discover_media_manifest(tmp_path, max_artifacts=4097)
