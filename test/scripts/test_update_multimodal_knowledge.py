# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from scripts.update_multimodal_knowledge import _extractor_identity


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "docs" / "architecture.svg").write_text(
        '<svg><text id="IndexCompiler">IndexCompiler</text></svg>',
        encoding="utf-8",
    )
    (repo / "README.md").write_text(
        "![IndexCompiler architecture](docs/architecture.svg)\n",
        encoding="utf-8",
    )
    (repo / "src" / "wiki.py").write_text(
        "class OtherService: pass\n",
        encoding="utf-8",
    )
    return repo


def _build(repo: Path, bundle: Path) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "scripts/build_multimodal_knowledge.py",
            str(repo),
            "--output",
            str(bundle),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def _update(repo: Path, bundle: Path, *options: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "scripts/update_multimodal_knowledge.py",
            str(repo),
            "--previous",
            str(bundle),
            *options,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _load(bundle: Path) -> dict:
    return json.loads(bundle.read_text(encoding="utf-8"))


def test_update_reuses_unchanged_fact_pack_and_preserves_mode(tmp_path):
    repo = _make_repo(tmp_path)
    bundle = tmp_path / "bundle.json"
    _build(repo, bundle)
    os.chmod(bundle, 0o640)

    completed = _update(repo, bundle)

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["unchanged"] == 1
    assert summary["reused_visual_fact_packs"] == 1
    assert summary["regenerated_visual_fact_packs"] == 0
    assert _load(bundle)["bundle_sha256"] == summary["bundle_sha256"]
    assert stat.S_IMODE(bundle.stat().st_mode) == 0o640


def test_update_regenerates_changed_media_and_drops_removed_media(tmp_path):
    repo = _make_repo(tmp_path)
    bundle = tmp_path / "bundle.json"
    _build(repo, bundle)
    previous_sha256 = _load(bundle)["bundle_sha256"]
    (repo / "docs" / "architecture.svg").write_text(
        '<svg><text id="QueryPlanner">QueryPlanner</text></svg>',
        encoding="utf-8",
    )

    changed = _update(repo, bundle)

    assert changed.returncode == 0, changed.stderr
    changed_summary = json.loads(changed.stdout)
    assert changed_summary["changed"] == 1
    assert changed_summary["regenerated_visual_fact_packs"] == 1
    assert changed_summary["bundle_sha256"] != previous_sha256

    (repo / "docs" / "architecture.svg").unlink()
    removed = _update(repo, bundle)

    assert removed.returncode == 0, removed.stderr
    removed_summary = json.loads(removed.stdout)
    assert removed_summary["removed"] == 1
    assert removed_summary["visual_fact_packs"] == 0


def test_update_adds_new_media_without_reextracting_unchanged_media(tmp_path):
    repo = _make_repo(tmp_path)
    bundle = tmp_path / "bundle.json"
    _build(repo, bundle)
    (repo / "docs" / "flow.svg").write_text(
        "<svg><text>RequestRouter</text></svg>", encoding="utf-8"
    )

    completed = _update(repo, bundle)

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["added"] == 1
    assert summary["unchanged"] == 1
    assert summary["reused_visual_fact_packs"] == 1
    assert summary["regenerated_visual_fact_packs"] == 1


def test_update_dry_run_does_not_modify_bundle(tmp_path):
    repo = _make_repo(tmp_path)
    bundle = tmp_path / "bundle.json"
    _build(repo, bundle)
    before = bundle.read_bytes()
    (repo / "docs" / "architecture.svg").write_text(
        "<svg><text>Changed</text></svg>", encoding="utf-8"
    )

    completed = _update(repo, bundle, "--dry-run")

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["dry_run"] is True
    assert summary["changed"] == 1
    assert summary["extract_artifacts"] == 1
    assert bundle.read_bytes() == before


def test_force_reextract_regenerates_unchanged_media(tmp_path):
    repo = _make_repo(tmp_path)
    bundle = tmp_path / "bundle.json"
    _build(repo, bundle)

    completed = _update(repo, bundle, "--force-reextract")

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["unchanged"] == 1
    assert summary["reused_visual_fact_packs"] == 0
    assert summary["regenerated_visual_fact_packs"] == 1


def test_code_only_change_recomputes_grounding_while_reusing_visual_facts(tmp_path):
    repo = _make_repo(tmp_path)
    bundle = tmp_path / "bundle.json"
    _build(repo, bundle)
    before = _load(bundle)
    assert before["grounding_manifest"]["binding_count"] == 0
    fact_sha256 = before["visual_facts_manifest"]["facts"][0]["fact_pack_sha256"]
    (repo / "src" / "wiki.py").write_text(
        "class IndexCompiler: pass\n",
        encoding="utf-8",
    )

    completed = _update(repo, bundle)

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    updated = _load(bundle)
    assert summary["reused_visual_fact_packs"] == 1
    assert summary["regenerated_visual_fact_packs"] == 0
    assert (
        updated["visual_facts_manifest"]["facts"][0]["fact_pack_sha256"] == fact_sha256
    )
    assert updated["grounding_manifest"]["binding_count"] > 0
    assert any(
        binding["symbol"] == "IndexCompiler"
        for binding in updated["grounding_manifest"]["bindings"]
    )


def test_update_can_write_a_separate_output(tmp_path):
    repo = _make_repo(tmp_path)
    previous = tmp_path / "previous.json"
    output = tmp_path / "updated.json"
    _build(repo, previous)
    before = previous.read_bytes()

    completed = _update(repo, previous, "--output", str(output))

    assert completed.returncode == 0, completed.stderr
    assert previous.read_bytes() == before
    assert _load(output)["schema"] == "codenib.multimodal-knowledge-bundle.v1"


def test_update_rejects_partial_vlm_config_without_overwriting(tmp_path):
    repo = _make_repo(tmp_path)
    bundle = tmp_path / "bundle.json"
    _build(repo, bundle)
    before = bundle.read_bytes()

    completed = _update(
        repo,
        bundle,
        "--visual-facts-model",
        "qwen-vl",
    )

    assert completed.returncode != 0
    assert "must be set together" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert bundle.read_bytes() == before


def test_update_rejects_invalid_previous_bundle_without_writing_output(tmp_path):
    repo = _make_repo(tmp_path)
    previous = tmp_path / "invalid.json"
    output = tmp_path / "updated.json"
    previous.write_text('{"schema":"wrong"}\n', encoding="utf-8")

    completed = _update(repo, previous, "--output", str(output))

    assert completed.returncode != 0
    assert "schema is unsupported" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not output.exists()


def test_extractor_identity_binds_provider_and_model_with_bounded_fallback():
    assert _extractor_identity("qwen", "qwen2.5-vl-72b") == ("qwen/qwen2.5-vl-72b")

    identity = _extractor_identity("供应商" * 30, "模型" * 80)

    assert len(identity.encode("utf-8")) <= 128
    assert "/sha256:" in identity
