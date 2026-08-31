# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.build_multimodal_knowledge import (
    _build_visual_grounding_scorer,
    build_parser,
)


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
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    counts = json.loads(completed.stdout)
    bundle = json.loads(output.read_text(encoding="utf-8"))
    assert counts["media_artifacts"] == 1
    assert counts["knowledge_entries"] == 1
    assert bundle["schema"] == "codenib.multimodal-knowledge-bundle.v1"
    assert len(bundle["bundle_sha256"]) == 64
    assert bundle["media_manifest"]["commit"] == "abc123"
    assert bundle["knowledge_view"]["entry_count"] == 1


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
        ]
    )

    assert args.visual_facts_model == "qwen-vl"
    assert args.visual_facts_api_base == "https://vlm.example/v1"
    assert args.visual_facts_api_key_env == "TEST_VLM_KEY"
    assert args.visual_facts_provider == "qwen"
    assert args.visual_facts_timeout == 15


def test_build_multimodal_knowledge_parser_accepts_index_grounding(tmp_path):
    output = tmp_path / "bundle.json"

    args = build_parser().parse_args(
        [
            str(tmp_path),
            "--output",
            str(output),
            "--grounding-indexes",
            "bm25+lsp",
            "--grounding-cache-dir",
            str(tmp_path / "indexes"),
            "--grounding-language",
            "python",
            "--grounding-language",
            "rust",
        ]
    )

    assert args.grounding_indexes == "bm25+lsp"
    assert args.grounding_cache_dir == str(tmp_path / "indexes")
    assert args.grounding_language == ["python", "rust"]


def test_build_multimodal_knowledge_wires_bm25_and_lsp_contexts(tmp_path, monkeypatch):
    calls = []
    bm25 = object()
    lsp = object()

    def fake_build(repo_path, skill_ids, **options):
        calls.append((repo_path, skill_ids, options))
        return {
            "retrieve": SimpleNamespace(bm25=bm25),
            "expand": SimpleNamespace(lsp_provider=lsp),
        }

    monkeypatch.setattr(
        "codenib.compiler.skill_context.build_skill_contexts",
        fake_build,
    )
    args = build_parser().parse_args(
        [
            str(tmp_path),
            "--output",
            str(tmp_path / "bundle.json"),
            "--grounding-indexes",
            "bm25+lsp",
            "--grounding-cache-dir",
            str(tmp_path / "indexes"),
            "--grounding-language",
            "rust",
        ]
    )

    scorer = _build_visual_grounding_scorer(args, repo_path=tmp_path)

    assert scorer.bm25 is bm25
    assert scorer.lsp_provider is lsp
    assert calls[0][1] == ["bm25_search", "lsp_definition", "lsp_references"]
    assert calls[0][2]["languages"] == ("rust",)
    assert calls[0][2]["cache_dir"] == str(tmp_path / "indexes")


def test_build_multimodal_knowledge_rejects_missing_requested_lsp(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "codenib.compiler.skill_context.build_skill_contexts",
        lambda *_args, **_kwargs: {
            "retrieve": SimpleNamespace(bm25=object()),
        },
    )
    args = build_parser().parse_args(
        [
            str(tmp_path),
            "--output",
            str(tmp_path / "bundle.json"),
            "--grounding-indexes",
            "bm25+lsp",
        ]
    )

    with pytest.raises(ValueError, match="did not load an LSP provider"):
        _build_visual_grounding_scorer(args, repo_path=tmp_path)


def test_build_multimodal_knowledge_wraps_loaded_graph_for_lsp(tmp_path, monkeypatch):
    graph = SimpleNamespace(
        name_to_vertex={},
        get_node_info_by_name=lambda _name: None,
    )
    monkeypatch.setattr(
        "codenib.compiler.skill_context.build_skill_contexts",
        lambda *_args, **_kwargs: {
            "retrieve": SimpleNamespace(bm25=object()),
            "expand": SimpleNamespace(lsp_provider=None, code_graph=graph),
        },
    )
    args = build_parser().parse_args(
        [
            str(tmp_path),
            "--output",
            str(tmp_path / "bundle.json"),
            "--grounding-indexes",
            "bm25+lsp",
        ]
    )

    scorer = _build_visual_grounding_scorer(args, repo_path=tmp_path)

    assert scorer.lsp_provider.graph is graph


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
