# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import json

from codenib.wiki.media_generation import (
    DeterministicSvgMediaGenerator,
    OpenAICompatibleImageGenerator,
    materialize_media_slots,
)


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_openai_compatible_image_generator_writes_asset(tmp_path):
    requests = []
    png = base64.b64encode(b"png-bytes").decode("ascii")

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return _Response({"data": [{"b64_json": png}]})

    generator = OpenAICompatibleImageGenerator(
        model="openai/gpt-image-1",
        api_base="http://media.local/v1",
        api_key="secret",
        size="512x512",
        timeout=12,
        urlopen=fake_urlopen,
    )
    asset = generator.generate(
        {
            "id": "overview-structure-diagram",
            "kind": "diagram",
            "purpose": "Explain the system map.",
            "prompt": "Create a compact architecture diagram.",
            "source_citations": ["src/app.py"],
        },
        output_dir=tmp_path,
    )

    assert asset["uri"] == "assets/wiki-media/overview-structure-diagram.png"
    assert asset["mime_type"] == "image/png"
    assert asset["model"] == "openai/gpt-image-1"
    assert asset["source_citations"] == ["src/app.py"]
    assert (tmp_path / "overview-structure-diagram.png").read_bytes() == b"png-bytes"
    request, timeout = requests[0]
    assert request.full_url == "http://media.local/v1/images/generations"
    assert request.get_header("Authorization") == "Bearer secret"
    assert timeout == 12
    body = json.loads(request.data.decode("utf-8"))
    assert body["model"] == "openai/gpt-image-1"
    assert body["size"] == "512x512"
    assert "src/app.py" in body["prompt"]


def test_materialize_media_slots_skips_unsupported_video_slots(tmp_path):
    calls = []

    def fake_urlopen(_request, timeout):
        calls.append(True)
        encoded = base64.b64encode(b"image").decode("ascii")
        return _Response({"data": [{"b64_json": encoded}]})

    generator = OpenAICompatibleImageGenerator(
        model="openai/gpt-image-1",
        api_base="http://media.local/v1",
        urlopen=fake_urlopen,
    )
    page = {
        "id": "overview",
        "media_slots": [
            {"id": "overview-image", "kind": "image", "prompt": "Draw it."},
            {"id": "overview-video", "kind": "video", "prompt": "Animate it."},
        ],
    }

    materialized = materialize_media_slots(
        page, generator=generator, output_dir=tmp_path
    )

    assert len(calls) == 1
    assert "asset" in materialized["media_slots"][0]
    assert "asset" not in materialized["media_slots"][1]


def test_deterministic_svg_generator_writes_visible_asset(tmp_path):
    generator = DeterministicSvgMediaGenerator()

    asset = generator.generate(
        {
            "id": "overview-image",
            "kind": "image",
            "title": "Overview image",
            "purpose": "Explain the repo visually.",
            "prompt": "Draw a source-grounded concept.",
            "source_citations": ["src/runtime.py"],
        },
        output_dir=tmp_path,
        asset_base_path="api/repos/demo/wiki-media/overview",
    )

    assert asset["uri"] == "api/repos/demo/wiki-media/overview/overview-image.svg"
    assert asset["mime_type"] == "image/svg+xml"
    assert asset["model"] == "local/svg"
    assert (
        (tmp_path / "overview-image.svg").read_text(encoding="utf-8").startswith("<svg")
    )


def test_media_generation_reuses_cached_asset(tmp_path):
    calls = []
    png = base64.b64encode(b"png-bytes").decode("ascii")

    def fake_urlopen(_request, timeout):
        calls.append(True)
        return _Response({"data": [{"b64_json": png}]})

    generator = OpenAICompatibleImageGenerator(
        model="openai/gpt-image-1",
        api_base="http://media.local/v1",
        urlopen=fake_urlopen,
    )
    slot = {
        "id": "overview-image",
        "kind": "image",
        "prompt": "Draw it.",
    }

    first = generator.generate(slot, output_dir=tmp_path)
    second = generator.generate(slot, output_dir=tmp_path)

    assert first == second
    assert len(calls) == 1
