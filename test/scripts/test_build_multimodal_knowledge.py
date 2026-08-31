# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys

from scripts.build_multimodal_knowledge import build_parser


def test_build_multimodal_knowledge_script_writes_bundle(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "docs").mkdir()
    (repo / "src").mkdir()
    (repo / "docs" / "architecture.svg").write_text(
        "<svg>WikiService</svg>",
        encoding="utf-8",
    )
    (repo / "README.md").write_text(
        "![WikiService architecture](docs/architecture.svg)",
        encoding="utf-8",
    )
    (repo / "src" / "wiki.py").write_text(
        "class WikiService: pass",
        encoding="utf-8",
    )
    generated = repo / "generated"
    generated.mkdir()
    (generated / "ignored.png").write_bytes(b"png")
    output = tmp_path / "bundle.json"
    graph_output = tmp_path / "visual-graphs.json"
    mermaid_dir = tmp_path / "mermaid"
    archify_dir = tmp_path / "archify"

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "scripts/build_multimodal_knowledge.py",
            str(repo),
            "--output",
            str(output),
            "--commit",
            "abc123",
            "--exclude-root",
            str(generated),
            "--visual-graph-output",
            str(graph_output),
            "--visual-graph-mermaid-dir",
            str(mermaid_dir),
            "--visual-graph-archify-dir",
            str(archify_dir),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    counts = json.loads(completed.stdout)
    bundle = json.loads(output.read_text(encoding="utf-8"))
    assert counts["media_artifacts"] == 1
    assert counts["knowledge_entries"] == 1
    assert counts["visual_graph_plans"] == 1
    assert bundle["schema"] == "codenib.multimodal-knowledge-bundle.v1"
    assert len(bundle["bundle_sha256"]) == 64
    assert bundle["media_manifest"]["commit"] == "abc123"
    assert bundle["knowledge_view"]["entry_count"] == 1
    graph_manifest = json.loads(graph_output.read_text(encoding="utf-8"))
    assert graph_manifest["schema"] == "codenib.visual-graph-manifest.v1"
    assert (
        graph_manifest["knowledge_view_sha256"]
        == bundle["knowledge_view"]["view_sha256"]
    )
    mermaid_files = list(mermaid_dir.glob("*.mmd"))
    assert len(mermaid_files) == 1
    mermaid = mermaid_files[0]
    assert "architecture.svg" in mermaid.name
    assert mermaid.read_text(encoding="utf-8").startswith("flowchart LR\n")
    archify_files = list(archify_dir.glob("*.architecture.json"))
    assert len(archify_files) == 1
    archify = json.loads(archify_files[0].read_text(encoding="utf-8"))
    assert archify["diagram_type"] == "architecture"
    assert len(archify["components"]) == len(graph_manifest["plans"][0]["nodes"])


def test_build_multimodal_knowledge_script_rejects_missing_repository(tmp_path):
    output = tmp_path / "bundle.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "scripts/build_multimodal_knowledge.py",
            str(tmp_path / "missing"),
            "--output",
            str(output),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode != 0
    assert "repository root does not exist" in completed.stderr
    assert not output.exists()


def test_build_multimodal_knowledge_parser_accepts_vlm_options(tmp_path):
    output = tmp_path / "bundle.json"

    args = build_parser().parse_args(
        [
            str(tmp_path),
            "--output",
            str(output),
            "--visual-facts-model",
            "qwen-vl",
            "--visual-facts-api-base",
            "https://vlm.example/v1",
            "--visual-facts-api-key-env",
            "TEST_VLM_KEY",
            "--visual-facts-provider",
            "qwen",
            "--visual-facts-timeout",
            "15",
            "--visual-graph-output",
            str(tmp_path / "graphs.json"),
            "--visual-graph-mermaid-dir",
            str(tmp_path / "mermaid"),
            "--visual-graph-archify-dir",
            str(tmp_path / "archify"),
            "--archify-repository-url",
            "https://github.com/sysevol-ai/CodeNib",
            "--archify-revision",
            "a" * 40,
        ]
    )

    assert args.visual_facts_model == "qwen-vl"
    assert args.visual_facts_api_base == "https://vlm.example/v1"
    assert args.visual_facts_api_key_env == "TEST_VLM_KEY"
    assert args.visual_facts_provider == "qwen"
    assert args.visual_facts_timeout == 15
    assert args.visual_graph_output == str(tmp_path / "graphs.json")
    assert args.visual_graph_mermaid_dir == str(tmp_path / "mermaid")
    assert args.visual_graph_archify_dir == str(tmp_path / "archify")
    assert args.archify_repository_url == "https://github.com/sysevol-ai/CodeNib"
    assert args.archify_revision == "a" * 40


def test_build_multimodal_knowledge_script_rejects_partial_vlm_config(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "bundle.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "scripts/build_multimodal_knowledge.py",
            str(repo),
            "--output",
            str(output),
            "--visual-facts-model",
            "qwen-vl",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode != 0
    assert "visual fact api_base is invalid" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not output.exists()
