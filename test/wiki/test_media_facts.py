# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from codenib.wiki.media_facts import (
    build_visual_fact_extraction_prompt,
    build_visual_facts_manifest,
    deterministic_visual_facts,
)


def _artifact():
    return {
        "path": "docs/assets/architecture.svg",
        "mime_type": "image/svg+xml",
        "sha256": "abc123",
        "role_hint": "architecture_diagram",
        "caption": "IndexCompiler to VectorStore architecture",
        "surrounding_text": "The diagram shows how IndexCompiler writes to VectorStore.",
        "references": [
            {
                "markdown_path": "README.md",
                "line": 7,
                "alt_text": "IndexCompiler to VectorStore architecture",
                "title": "",
            }
        ],
    }


def test_build_visual_fact_extraction_prompt_requests_structured_json():
    prompt = build_visual_fact_extraction_prompt(_artifact())

    assert "extract_repository_visual_facts" in prompt
    assert "entities" in prompt
    assert "relations" in prompt
    assert "grounding_candidates" in prompt
    assert "IndexCompiler" in prompt
    assert "Return JSON only" in prompt


def test_deterministic_visual_facts_extracts_metadata_entities_and_claims():
    facts = deterministic_visual_facts(_artifact())

    assert facts["artifact_path"] == "docs/assets/architecture.svg"
    assert facts["artifact_sha256"] == "abc123"
    assert facts["role_hint"] == "architecture_diagram"
    assert facts["extractor"] == "local/metadata"
    entity_names = {entity["name"] for entity in facts["entities"]}
    assert "IndexCompiler" in entity_names
    assert "VectorStore" in entity_names
    assert facts["claims"]
    assert facts["fact_pack_sha256"]


def test_build_visual_facts_manifest_is_stable_and_links_media_manifest():
    media_manifest = {
        "manifest_sha256": "media-manifest-hash",
        "artifacts": [_artifact()],
    }

    first = build_visual_facts_manifest(media_manifest)
    second = build_visual_facts_manifest(media_manifest)

    assert first == second
    assert first["schema"] == "codenib.media-facts.v1"
    assert first["media_manifest_sha256"] == "media-manifest-hash"
    assert first["fact_count"] == 1
    assert first["manifest_sha256"]


def test_build_visual_facts_manifest_accepts_custom_vlm_extractor():
    media_manifest = {
        "manifest_sha256": "media-manifest-hash",
        "artifacts": [_artifact()],
    }

    def fake_vlm_extractor(artifact):
        return {
            "artifact_path": artifact["path"],
            "artifact_sha256": artifact["sha256"],
            "role_hint": artifact["role_hint"],
            "extractor": "vlm/test",
            "entities": [
                {
                    "name": "IndexCompiler",
                    "type": "component",
                    "evidence": "visual label",
                    "confidence": 0.9,
                    "grounding_candidates": ["IndexCompiler"],
                }
            ],
            "relations": [
                {
                    "source": "IndexCompiler",
                    "relation": "writes_to",
                    "target": "VectorStore",
                    "evidence": "arrow",
                    "confidence": 0.8,
                }
            ],
            "claims": [
                {
                    "text": "IndexCompiler writes to VectorStore.",
                    "evidence": "arrow label",
                    "confidence": 0.8,
                }
            ],
            "metadata": {"provider": "fake"},
        }

    manifest = build_visual_facts_manifest(
        media_manifest,
        extractor=fake_vlm_extractor,
    )

    fact = manifest["facts"][0]
    assert fact["extractor"] == "vlm/test"
    assert fact["relations"] == [
        {
            "source": "IndexCompiler",
            "relation": "writes_to",
            "target": "VectorStore",
            "evidence": "arrow",
            "confidence": 0.8,
        }
    ]


def test_build_visual_fact_extraction_prompt_bounds_oversized_context():
    artifact = _artifact()
    artifact["surrounding_text"] = "x" * (40 * 1024)

    prompt = build_visual_fact_extraction_prompt(artifact)

    assert len(prompt.encode("utf-8")) < 32 * 1024
    assert "\\u2026" in prompt


def test_fact_manifest_output_is_json_serializable():
    payload = build_visual_facts_manifest(
        {"manifest_sha256": "media-manifest-hash", "artifacts": [_artifact()]}
    )

    json.dumps(payload, allow_nan=False)
