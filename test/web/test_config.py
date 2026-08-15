# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from codenib.web.config import load_config


def test_minimal_config_uses_concrete_dataclass_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("wiki_agent: false\n")

    config = load_config(str(config_path))

    assert config.model == "gpt-4o"
    assert config.mode == "sparse"
    assert config.embedding_dimension == 384
    assert config.max_turns == 8
    assert config.wiki_agent is False


def test_wiki_media_config_enables_local_renderer_without_endpoint(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
wiki_media_model: local/svg
wiki_media_options:
  provider: local
  width: 1024
""".lstrip()
    )

    config = load_config(str(config_path))

    assert config.wiki_media_generation_enabled is True
    assert config.wiki_media_api_base is None
    assert config.wiki_media_options == {"provider": "local", "width": 1024}


def test_wiki_media_environment_overrides_file_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
wiki_media_model: local/svg
wiki_media_options:
  provider: local
""".lstrip()
    )
    monkeypatch.setenv("CODENIB_WIKI_MEDIA_MODEL", "openai/image-1")
    monkeypatch.setenv("CODENIB_WIKI_MEDIA_API_BASE", "https://images.example/v1")
    monkeypatch.setenv("CODENIB_WIKI_MEDIA_API_KEY", "secret")
    monkeypatch.setenv(
        "CODENIB_WIKI_MEDIA_OPTIONS",
        '{"provider":"openai","timeout":30}',
    )

    config = load_config(str(config_path))

    assert config.wiki_media_generation_enabled is True
    assert config.wiki_media_model == "openai/image-1"
    assert config.wiki_media_api_base == "https://images.example/v1"
    assert config.wiki_media_api_key == "secret"
    assert config.wiki_media_options == {
        "provider": "openai",
        "timeout": 30,
    }
