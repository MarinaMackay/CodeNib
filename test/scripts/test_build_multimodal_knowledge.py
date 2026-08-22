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
    output = tmp_path / "bundle.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_multimodal_knowledge.py",
            str(repo),
            "--output",
            str(output),
            "--commit",
            "abc123",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    counts = json.loads(completed.stdout)
    bundle = json.loads(output.read_text(encoding="utf-8"))
    assert counts["media_artifacts"] == 1
    assert counts["knowledge_entries"] == 1
    assert bundle["media_manifest"]["commit"] == "abc123"
    assert bundle["knowledge_view"]["entry_count"] == 1


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
