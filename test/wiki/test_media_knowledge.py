# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from codenib.wiki.media_knowledge import (
    MultimodalKnowledgeView,
    build_multimodal_knowledge_view,
    find_visual_code_links,
    get_visual_evidence,
    search_visual_context,
)


def _view():
    media = {
        "manifest_sha256": "media-hash",
        "artifacts": [
            {
                "path": "docs/assets/architecture.svg",
                "role_hint": "architecture_diagram",
                "caption": "IndexCompiler architecture",
                "surrounding_text": "IndexCompiler writes to VectorStore.",
            }
        ],
    }
    facts = {
        "manifest_sha256": "facts-hash",
        "facts": [
            {
                "artifact_path": "docs/assets/architecture.svg",
                "entities": [
                    {
                        "name": "IndexCompiler",
                        "type": "component",
                        "evidence": "caption",
                    },
                    {
                        "name": "VectorStore",
                        "type": "component",
                        "evidence": "caption",
                    },
                ],
                "claims": [
                    {
                        "text": "IndexCompiler writes to VectorStore.",
                        "evidence": "surrounding markdown",
                    }
                ],
            }
        ],
    }
    grounding = {
        "manifest_sha256": "grounding-hash",
        "bindings": [
            {
                "artifact_path": "docs/assets/architecture.svg",
                "entity_name": "IndexCompiler",
                "source_path": "codenib/compiler/index_compiler.py",
                "symbol": "IndexCompiler",
                "kind": "symbol",
                "line": 42,
                "score": 1.0,
                "evidence": "exact symbol match",
            }
        ],
    }
    return build_multimodal_knowledge_view(media, facts, grounding)


def test_build_multimodal_knowledge_view_joins_artifacts_facts_and_bindings():
    view = _view()

    assert view["schema"] == "codenib.multimodal-knowledge-view.v1"
    assert view["media_manifest_sha256"] == "media-hash"
    assert view["visual_facts_manifest_sha256"] == "facts-hash"
    assert view["grounding_manifest_sha256"] == "grounding-hash"
    assert view["entry_count"] == 1
    assert view["entries"][0]["artifact"]["path"] == "docs/assets/architecture.svg"
    assert view["entries"][0]["facts"]["entities"][0]["name"] == "IndexCompiler"
    assert view["entries"][0]["bindings"][0]["symbol"] == "IndexCompiler"
    assert view["view_sha256"]


def test_search_visual_context_finds_visual_entries():
    results = search_visual_context(_view(), "VectorStore architecture")

    assert len(results) == 1
    assert results[0]["artifact_path"] == "docs/assets/architecture.svg"
    assert results[0]["score"] >= 2


def test_get_visual_evidence_returns_one_artifact_entry():
    evidence = get_visual_evidence(_view(), "docs/assets/architecture.svg")

    assert evidence is not None
    assert evidence["artifact"]["caption"] == "IndexCompiler architecture"
    assert (
        evidence["bindings"][0]["source_path"] == "codenib/compiler/index_compiler.py"
    )


def test_find_visual_code_links_returns_entries_for_source_symbol():
    links = find_visual_code_links(
        _view(),
        "codenib/compiler/index_compiler.py",
        symbol="IndexCompiler",
    )

    assert len(links) == 1
    assert links[0]["artifact_path"] == "docs/assets/architecture.svg"
    assert links[0]["binding"]["line"] == 42


def test_multimodal_knowledge_view_wrapper_methods():
    wrapper = MultimodalKnowledgeView(_view())

    assert wrapper.search_visual_context("IndexCompiler")
    assert wrapper.get_visual_evidence("docs/assets/architecture.svg") is not None
    assert wrapper.find_visual_code_links("codenib/compiler/index_compiler.py")
