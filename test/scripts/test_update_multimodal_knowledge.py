# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys


def test_update_multimodal_knowledge_script_reuses_unchanged_facts(tmp_path):
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "docs" / "renderer.svg").write_text(
        "<svg><text>WikiRenderer</text></svg>",
        encoding="utf-8",
    )
    (repo / "README.md").write_text(
        "![WikiRenderer](docs/renderer.svg)\n",
        encoding="utf-8",
    )
    (repo / "src" / "wiki.py").write_text(
        "class WikiRenderer: pass\n",
        encoding="utf-8",
    )
    previous = tmp_path / "previous.json"
    updated = tmp_path / "updated.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_multimodal_knowledge.py",
            str(repo),
            "--output",
            str(previous),
            "--commit",
            "old",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    (repo / "docs" / "store.svg").write_text(
        "<svg><text>VectorStore</text></svg>",
        encoding="utf-8",
    )
    (repo / "README.md").write_text(
        "\n".join(
            [
                "![WikiRenderer](docs/renderer.svg)",
                "![VectorStore](docs/store.svg)",
            ]
        ),
        encoding="utf-8",
    )
    (repo / "src" / "store.py").write_text(
        "class VectorStore: pass\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/update_multimodal_knowledge.py",
            str(repo),
            "--previous",
            str(previous),
            "--output",
            str(updated),
            "--commit",
            "new",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    summary = json.loads(completed.stdout)
    bundle = json.loads(updated.read_text(encoding="utf-8"))
    fact_paths = {
        fact["artifact_path"] for fact in bundle["visual_facts_manifest"]["facts"]
    }

    assert summary["added"] == 1
    assert summary["unchanged"] == 1
    assert summary["reused_visual_fact_packs"] == 1
    assert summary["regenerated_visual_fact_packs"] == 1
    assert summary["visual_fact_packs"] == 2
    assert summary["visual_graph_plans"] == 2
    assert fact_paths == {"docs/renderer.svg", "docs/store.svg"}
    assert bundle["visual_graph_manifest"]["plan_count"] == 2
    assert bundle["incremental_update"]["extract_artifact_paths"] == ["docs/store.svg"]
