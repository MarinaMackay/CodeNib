# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible VLM extraction for repository media artifacts."""

from __future__ import annotations

import base64
import json
import math
import os
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .media_facts import (
    build_visual_fact_extraction_prompt,
    normalize_visual_fact_pack,
)
from .media_graph_plan import validate_visual_graph_plan

_MAX_MODEL_LENGTH = 256
_MAX_PROVIDER_LENGTH = 128
_MAX_URL_LENGTH = 4096
_MAX_TIMEOUT_SECONDS = 600.0
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_IMAGE_BYTES = 16 * 1024 * 1024
_SUPPORTED_MIME_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/svg+xml",
        "image/webp",
    }
)


class OpenAICompatibleVisualFactExtractor:
    """Extract structured visual facts through an OpenAI-compatible chat API."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        urlopen: Callable[..., Any] | None = None,
        provider: str = "openai-compatible",
    ) -> None:
        self.model = str(model or "").strip()
        self.api_base = str(api_base or "").strip()
        self.api_key = _validated_api_key(api_key)
        self.timeout = _validated_timeout(timeout)
        self.provider = str(provider or "").strip()
        self._urlopen = urlopen or urllib.request.urlopen
        if not self.model:
            raise ValueError("visual fact model is required")
        if len(self.model) > _MAX_MODEL_LENGTH:
            raise ValueError("visual fact model is too long")
        if not self.provider or len(self.provider) > _MAX_PROVIDER_LENGTH:
            raise ValueError("visual fact provider is invalid")
        self._endpoint = _chat_completions_endpoint(self.api_base)

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def extract(
        self,
        artifact: Mapping[str, Any],
        *,
        repo_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Extract one canonical visual fact pack for *artifact*."""

        prompt = build_visual_fact_extraction_prompt(artifact)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if repo_path is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _artifact_data_url(repo_path, artifact),
                    },
                }
            )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract structured repository visual facts. "
                        "Return JSON only."
                    ),
                },
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        response = self._post_json(payload)
        extracted = _response_content_json(response)
        extracted.setdefault("artifact_path", str(artifact.get("path") or ""))
        extracted.setdefault("artifact_sha256", str(artifact.get("sha256") or ""))
        extracted.setdefault("role_hint", str(artifact.get("role_hint") or ""))
        extracted["extractor"] = self.provider
        metadata = dict(extracted.get("metadata") or {})
        metadata.update({"model": self.model, "provider": self.provider})
        extracted["metadata"] = metadata
        return normalize_visual_fact_pack(extracted)

    def __call__(self, artifact: Mapping[str, Any]) -> dict[str, Any]:
        return self.extract(artifact)

    def _post_json(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with self._urlopen(request, timeout=self.timeout) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("visual fact response exceeds the byte limit")
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("visual fact response must be a JSON object")
        return data


class OpenAICompatibleVisualGraphPlanExtractor:
    """Plan a validated visual graph through an OpenAI-compatible VLM API."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        urlopen: Callable[..., Any] | None = None,
        provider: str = "openai-compatible",
    ) -> None:
        self.model = str(model or "").strip()
        self.api_base = str(api_base or "").strip()
        self.api_key = _validated_api_key(api_key)
        self.timeout = _validated_timeout(timeout)
        self.provider = str(provider or "").strip()
        self._urlopen = urlopen or urllib.request.urlopen
        if not self.model:
            raise ValueError("visual graph plan model is required")
        if len(self.model) > _MAX_MODEL_LENGTH:
            raise ValueError("visual graph plan model is too long")
        if not self.provider or len(self.provider) > _MAX_PROVIDER_LENGTH:
            raise ValueError("visual graph plan provider is invalid")
        self._endpoint = _chat_completions_endpoint(
            self.api_base,
            label="visual graph plan api_base",
        )

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def plan(
        self,
        entry: Mapping[str, Any],
        *,
        repo_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Return one validated visual graph plan for a knowledge-view entry."""

        artifact = entry.get("artifact") or {}
        prompt = build_visual_graph_plan_prompt(entry)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if repo_path is not None and isinstance(artifact, Mapping):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _artifact_data_url(repo_path, artifact),
                    },
                }
            )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You produce validated repository visual graph plans. "
                        "Return JSON only."
                    ),
                },
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        response = self._post_json(payload)
        extracted = _response_content_json(response)
        plan = extracted.get("graph_plan") if isinstance(extracted, Mapping) else None
        if not isinstance(plan, Mapping):
            plan = extracted
        plan = dict(plan)
        if not plan.get("artifact_path") and isinstance(artifact, Mapping):
            plan["artifact_path"] = str(artifact.get("path") or "")
        normalized = validate_visual_graph_plan(plan)
        metadata = dict(normalized.get("metadata") or {})
        metadata.update({"model": self.model, "provider": self.provider})
        normalized["metadata"] = metadata
        return normalized

    def __call__(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        return self.plan(entry)

    def _post_json(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with self._urlopen(request, timeout=self.timeout) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("visual graph plan response exceeds the byte limit")
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("visual graph plan response must be a JSON object")
        return data


def build_visual_graph_plan_prompt(entry: Mapping[str, Any]) -> str:
    """Build a bounded prompt for model-produced visual graph plans."""

    artifact = entry.get("artifact") or {}
    facts = entry.get("facts") or {}
    bindings = entry.get("bindings") or []
    payload = {
        "artifact": _bounded_jsonable(artifact),
        "facts": _bounded_jsonable(facts),
        "bindings": _bounded_jsonable(list(bindings)[:24]),
        "required_schema": {
            "schema": "codenib.visual-graph-plan.v1",
            "version": 1,
            "artifact_path": "repository-relative visual artifact path",
            "nodes": [
                {
                    "id": "stable_ascii_id",
                    "label": "reader-facing label",
                    "source_path": "repository-relative source path when known",
                    "symbol": "source symbol when known",
                    "line": 0,
                    "evidence": "why this node is grounded",
                }
            ],
            "edges": [
                {
                    "source": "node id",
                    "target": "node id",
                    "relation": "calls|references|contains|depends_on|related_to",
                    "evidence": "visual/source evidence for the edge",
                }
            ],
        },
    }
    return (
        "Create a compact source-grounded graph plan for this repository visual. "
        "Use only entities and source bindings present in the input. Keep node "
        "ids unique, reference only existing node ids in edges, keep paths "
        "repository-relative, and return JSON only.\n\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def visual_fact_extractor_from_config(
    config: Any,
) -> OpenAICompatibleVisualFactExtractor | None:
    """Build a visual-fact extractor from ``QAConfig``-shaped settings."""

    if not bool(getattr(config, "wiki_visual_fact_extraction_enabled", False)):
        return None
    model = str(getattr(config, "wiki_visual_facts_model", None) or "").strip()
    api_base = str(getattr(config, "wiki_visual_facts_api_base", None) or "").strip()
    options = dict(getattr(config, "wiki_visual_facts_options", {}) or {})
    timeout = options.get("timeout", 120.0)
    provider = str(options.get("provider") or "openai-compatible")
    return OpenAICompatibleVisualFactExtractor(
        model=model,
        api_base=api_base,
        api_key=getattr(config, "wiki_visual_facts_api_key", None),
        timeout=timeout,
        provider=provider,
    )


def visual_graph_planner_from_config(
    config: Any,
) -> OpenAICompatibleVisualGraphPlanExtractor | None:
    """Build a visual graph planner from ``QAConfig``-shaped settings."""

    if not bool(getattr(config, "wiki_visual_graph_planning_enabled", False)):
        return None
    model = str(getattr(config, "wiki_visual_graph_model", None) or "").strip()
    api_base = str(getattr(config, "wiki_visual_graph_api_base", None) or "").strip()
    options = dict(getattr(config, "wiki_visual_graph_options", {}) or {})
    timeout = options.get("timeout", 120.0)
    provider = str(options.get("provider") or "openai-compatible")
    return OpenAICompatibleVisualGraphPlanExtractor(
        model=model,
        api_base=api_base,
        api_key=getattr(config, "wiki_visual_graph_api_key", None),
        timeout=timeout,
        provider=provider,
    )


def _bounded_jsonable(value: Any, *, max_bytes: int = 32_768) -> Any:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    text = encoded[:max_bytes].decode("utf-8", "ignore")
    return {"truncated_json": text}


def _response_content_json(response: Mapping[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("visual fact response must contain choices[0]")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("visual fact response choice must be an object")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("visual fact response choice must contain message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("visual fact response message content must be a string")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("visual fact response content must decode to a JSON object")
    return parsed


def _artifact_data_url(repo_path: str | Path, artifact: Mapping[str, Any]) -> str:
    root = Path(repo_path).expanduser().resolve()
    relative = _safe_relative_path(artifact.get("path"))
    path = (root / relative).resolve()
    if not (path == root or root in path.parents):
        raise ValueError("visual artifact path is outside the repository")
    if path.is_symlink() or not path.is_file():
        raise ValueError("visual artifact must be a regular file")
    mime_type = str(artifact.get("mime_type") or "").strip()
    if mime_type not in _SUPPORTED_MIME_TYPES:
        raise ValueError("visual artifact MIME type is unsupported")
    size = path.stat().st_size
    if size < 0 or size > _MAX_IMAGE_BYTES:
        raise ValueError("visual artifact exceeds the byte limit")
    data = path.read_bytes()
    if len(data) > _MAX_IMAGE_BYTES:
        raise ValueError("visual artifact exceeds the byte limit")
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def _safe_relative_path(value: Any) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("visual artifact path is required")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise ValueError("visual artifact path must be repository-relative")
    return Path(*path.parts)


def _validated_timeout(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("visual fact timeout must be a positive number")
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("visual fact timeout must be a positive number") from exc
    if not math.isfinite(timeout) or not 0 < timeout <= _MAX_TIMEOUT_SECONDS:
        raise ValueError(
            f"visual fact timeout must be between 0 and {_MAX_TIMEOUT_SECONDS:g} seconds"
        )
    return timeout


def _validated_api_key(value: Any) -> str | None:
    if value is None:
        return None
    api_key = str(value)
    if len(api_key) > 8192 or any(
        ord(character) < 0x20 or ord(character) == 0x7F for character in api_key
    ):
        raise ValueError("visual fact API key is invalid")
    return api_key


def _validated_http_url(value: str, *, label: str) -> str:
    url = str(value or "").strip()
    if not url or len(url) > _MAX_URL_LENGTH:
        raise ValueError(f"{label} is invalid")
    if any(
        character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F
        for character in url
    ):
        raise ValueError(f"{label} is invalid")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        raise ValueError(f"{label} must be an HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not contain credentials")
    return url


def _chat_completions_endpoint(
    api_base: str,
    *,
    label: str = "visual fact api_base",
) -> str:
    base = _validated_http_url(api_base, label=label)
    parsed = urlsplit(base)
    path = parsed.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path = f"{path}/chat/completions"
    return parsed._replace(path=path, fragment="").geturl()


__all__ = [
    "OpenAICompatibleVisualFactExtractor",
    "OpenAICompatibleVisualGraphPlanExtractor",
    "build_visual_graph_plan_prompt",
    "visual_fact_extractor_from_config",
    "visual_graph_planner_from_config",
]
