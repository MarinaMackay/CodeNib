# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys


def test_smoke_multimodal_vlm_script_writes_demo_bundle(tmp_path):
    output = tmp_path / "smoke-bundle.json"
    repo = tmp_path / "smoke-repo"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_multimodal_vlm.py",
            "--output",
            str(output),
            "--keep-repo",
            str(repo),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    counts = json.loads(completed.stdout)
    bundle = json.loads(output.read_text(encoding="utf-8"))
    fact = bundle["visual_facts_manifest"]["facts"][0]
    entity_names = {entity["name"] for entity in fact["entities"]}

    assert counts["extractor"] == "local/metadata"
    assert counts["media_artifacts"] == 1
    assert counts["knowledge_entries"] == 1
    assert bundle["schema"] == "codenib.multimodal-knowledge-bundle.v1"
    assert "WikiRenderer" in entity_names
    assert "IndexCompiler" in entity_names
    assert "VectorStore" in entity_names
