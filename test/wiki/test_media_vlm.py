# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import codenib.wiki.media_vlm as media_vlm
from codenib.wiki.media_vlm import (
    OpenAICompatibleVisualFactExtractor,
    OpenAICompatibleVisualGraphPlanExtractor,
    build_visual_graph_plan_prompt,
    visual_fact_extractor_from_config,
    visual_graph_planner_from_config,
)


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def read(self, size: int = -1) -> bytes:
        payload = json.dumps(self._payload).encode("utf-8")
        return payload if size < 0 else payload[:size]


def _artifact():
    return {
        "path": "docs/assets/architecture.png",
        "mime_type": "image/png",
        "sha256": "abc123",
        "role_hint": "architecture_diagram",
        "caption": "IndexCompiler architecture",
        "surrounding_text": "IndexCompiler writes to VectorStore.",
    }


def test_openai_compatible_visual_fact_extractor_posts_image_and_normalizes(tmp_path):
    (tmp_path / "docs" / "assets").mkdir(parents=True)
    (tmp_path / "docs" / "assets" / "architecture.png").write_bytes(b"png-bytes")
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
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
                                            "evidence": "arrow",
                                            "confidence": 0.8,
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            }
        )

    extractor = OpenAICompatibleVisualFactExtractor(
        model="qwen-vl",
        api_base="https://api.example/v1",
        api_key="secret",
        timeout=7,
        urlopen=fake_urlopen,
    )
    facts = extractor.extract(_artifact(), repo_path=tmp_path)

    assert facts["artifact_path"] == "docs/assets/architecture.png"
    assert facts["artifact_sha256"] == "abc123"
    assert facts["extractor"] == "openai-compatible"
    assert facts["entities"][0]["name"] == "IndexCompiler"
    assert facts["metadata"]["model"] == "qwen-vl"
    request, timeout = requests[0]
    assert request.full_url == "https://api.example/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer secret"
    assert timeout == 7
    body = json.loads(request.data.decode("utf-8"))
    assert body["model"] == "qwen-vl"
    content = body["messages"][1]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"api_base": "file:///tmp"}, "api_base"),
        ({"api_base": "https://user:secret@example.test/v1"}, "credentials"),
        ({"api_key": "secret\nheader"}, "API key"),
        ({"timeout": True}, "positive number"),
        ({"timeout": 601}, "between 0"),
    ],
)
def test_visual_fact_extractor_rejects_unsafe_configuration(overrides, message):
    arguments = {"model": "qwen-vl", "api_base": "https://api.example/v1"}
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        OpenAICompatibleVisualFactExtractor(**arguments)


def test_visual_fact_extractor_rejects_unsafe_artifact_path(tmp_path):
    extractor = OpenAICompatibleVisualFactExtractor(
        model="qwen-vl",
        api_base="https://api.example/v1",
        urlopen=lambda _request, timeout: _Response({"choices": []}),
    )
    artifact = {**_artifact(), "path": "../secret.png"}

    with pytest.raises(ValueError, match="repository-relative"):
        extractor.extract(artifact, repo_path=tmp_path)


def test_visual_fact_extractor_rejects_unsupported_mime_type(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "asset.gif").write_bytes(b"gif")
    extractor = OpenAICompatibleVisualFactExtractor(
        model="qwen-vl",
        api_base="https://api.example/v1",
        urlopen=lambda _request, timeout: _Response({"choices": []}),
    )

    with pytest.raises(ValueError, match="MIME"):
        extractor.extract(
            {
                "path": "docs/asset.gif",
                "mime_type": "image/gif",
                "sha256": "abc123",
            },
            repo_path=tmp_path,
        )


def test_visual_fact_extractor_bounds_response_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(media_vlm, "_MAX_RESPONSE_BYTES", 16)

    class OversizedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def read(self, size):
            assert size == 17
            return b"x" * size

    extractor = OpenAICompatibleVisualFactExtractor(
        model="qwen-vl",
        api_base="https://api.example/v1",
        urlopen=lambda _request, timeout: OversizedResponse(),
    )

    with pytest.raises(ValueError, match="response exceeds"):
        extractor.extract(_artifact())


def test_visual_fact_extractor_rejects_non_json_content():
    extractor = OpenAICompatibleVisualFactExtractor(
        model="qwen-vl",
        api_base="https://api.example/v1",
        urlopen=lambda _request, timeout: _Response(
            {"choices": [{"message": {"content": "not-json"}}]}
        ),
    )

    with pytest.raises(json.JSONDecodeError):
        extractor.extract(_artifact())


def test_visual_fact_extractor_from_config_returns_none_when_disabled():
    config = SimpleNamespace(wiki_visual_fact_extraction_enabled=False)

    assert visual_fact_extractor_from_config(config) is None


def test_visual_fact_extractor_from_config_builds_provider():
    config = SimpleNamespace(
        wiki_visual_fact_extraction_enabled=True,
        wiki_visual_facts_model="qwen-vl",
        wiki_visual_facts_api_base="https://vlm.example/v1",
        wiki_visual_facts_api_key="secret",
        wiki_visual_facts_options={
            "provider": "qwen",
            "timeout": 9,
        },
    )

    extractor = visual_fact_extractor_from_config(config)

    assert isinstance(extractor, OpenAICompatibleVisualFactExtractor)
    assert extractor.model == "qwen-vl"
    assert extractor.endpoint == "https://vlm.example/v1/chat/completions"
    assert extractor.provider == "qwen"
    assert extractor.timeout == 9


def _entry():
    return {
        "artifact": {
            "path": "docs/assets/architecture.png",
            "mime_type": "image/png",
            "sha256": "abc123",
            "caption": "WikiRenderer architecture",
        },
        "facts": {
            "entities": [
                {"name": "WikiRenderer", "type": "component"},
                {"name": "IndexCompiler", "type": "component"},
            ],
        },
        "bindings": [
            {
                "entity_name": "WikiRenderer",
                "source_path": "src/wiki.py",
                "symbol": "WikiRenderer",
                "line": 1,
            },
            {
                "entity_name": "IndexCompiler",
                "source_path": "src/compiler.py",
                "symbol": "IndexCompiler",
                "line": 7,
            },
        ],
    }


def test_openai_compatible_visual_graph_planner_posts_image_and_validates(tmp_path):
    (tmp_path / "docs" / "assets").mkdir(parents=True)
    (tmp_path / "docs" / "assets" / "architecture.png").write_bytes(b"png-bytes")
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "graph_plan": {
                                        "schema": "codenib.visual-graph-plan.v1",
                                        "version": 1,
                                        "nodes": [
                                            {
                                                "id": "WikiRenderer",
                                                "label": "WikiRenderer",
                                                "source_path": "src/wiki.py",
                                                "symbol": "WikiRenderer",
                                                "line": 1,
                                            },
                                            {
                                                "id": "IndexCompiler",
                                                "label": "IndexCompiler",
                                                "source_path": "src/compiler.py",
                                                "symbol": "IndexCompiler",
                                                "line": 7,
                                            },
                                        ],
                                        "edges": [
                                            {
                                                "source": "WikiRenderer",
                                                "target": "IndexCompiler",
                                                "relation": "calls",
                                            }
                                        ],
                                    }
                                }
                            )
                        }
                    }
                ]
            }
        )

    planner = OpenAICompatibleVisualGraphPlanExtractor(
        model="qwen-vl",
        api_base="https://api.example/v1",
        api_key="secret",
        timeout=8,
        urlopen=fake_urlopen,
        provider="qwen",
    )
    plan = planner.plan(_entry(), repo_path=tmp_path)

    assert plan["artifact_path"] == "docs/assets/architecture.png"
    assert plan["nodes"][0]["label"] == "WikiRenderer"
    assert plan["edges"][0]["relation"] == "calls"
    assert plan["metadata"] == {"model": "qwen-vl", "provider": "qwen"}
    request, timeout = requests[0]
    assert request.full_url == "https://api.example/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer secret"
    assert timeout == 8
    body = json.loads(request.data.decode("utf-8"))
    assert "required_schema" in body["messages"][1]["content"][0]["text"]
    assert body["messages"][1]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_visual_graph_planner_rejects_invalid_model_plan():
    planner = OpenAICompatibleVisualGraphPlanExtractor(
        model="qwen-vl",
        api_base="https://api.example/v1",
        urlopen=lambda _request, timeout: _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "graph_plan": {
                                        "schema": "codenib.visual-graph-plan.v1",
                                        "version": 1,
                                        "artifact_path": "docs/architecture.svg",
                                        "nodes": [{"id": "A", "label": "A"}],
                                        "edges": [{"source": "A", "target": "Missing"}],
                                    }
                                }
                            )
                        }
                    }
                ]
            }
        ),
    )

    with pytest.raises(ValueError, match="endpoints"):
        planner.plan(_entry())


def test_visual_graph_plan_prompt_contains_grounding_contract():
    prompt = build_visual_graph_plan_prompt(_entry())

    assert "Use only entities and source bindings" in prompt
    assert "codenib.visual-graph-plan.v1" in prompt
    assert "src/wiki.py" in prompt


def test_visual_graph_planner_from_config_builds_provider():
    config = SimpleNamespace(
        wiki_visual_graph_planning_enabled=True,
        wiki_visual_graph_model="qwen-vl",
        wiki_visual_graph_api_base="https://vlm.example/v1",
        wiki_visual_graph_api_key="secret",
        wiki_visual_graph_options={
            "provider": "qwen",
            "timeout": 9,
        },
    )

    planner = visual_graph_planner_from_config(config)

    assert isinstance(planner, OpenAICompatibleVisualGraphPlanExtractor)
    assert planner.model == "qwen-vl"
    assert planner.endpoint == "https://vlm.example/v1/chat/completions"
    assert planner.provider == "qwen"
    assert planner.timeout == 9
