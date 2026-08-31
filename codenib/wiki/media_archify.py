# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Compile validated visual graph plans to Archify architecture IR."""

from __future__ import annotations

import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .media_graph_plan import validate_visual_graph_plan

ARCHIFY_SCHEMA_VERSION = 1
ARCHIFY_DIAGRAM_TYPE = "architecture"

_GITHUB_REPOSITORY_RE = re.compile(
    r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?/?"
)
_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}")
_MAX_TITLE_BYTES = 512
_MAX_SOURCE_PATH_BYTES = 240
_MAX_SOURCE_LABEL_BYTES = 48
_MAX_OUTPUT_BYTES = 8 * 1024 * 1024


def compile_visual_graph_plan_to_archify(
    plan: Mapping[str, Any],
    *,
    title: str | None = None,
    repository_url: str | None = None,
    revision: str | None = None,
    minimum_grounding_score: float = 0.8,
) -> dict[str, Any]:
    """Return Archify v1 architecture JSON from one validated graph plan.

    Repository evidence is opt-in and fail-closed: URL and full revision must
    be supplied together. Only source bindings meeting the declared grounding
    threshold are exported, while graph edges remain limited to relations that
    already passed VisualGraphPlan validation.
    """

    normalized = validate_visual_graph_plan(plan)
    threshold = _grounding_threshold(minimum_grounding_score)
    repository = _repository_identity(repository_url, revision)
    document_title = _title(title or _default_title(normalized["artifact_path"]))

    components = []
    for index, node in enumerate(normalized["nodes"]):
        component: dict[str, Any] = {
            "id": node["id"],
            "type": "backend",
            "label": node["label"],
            "pos": [80 + (index % 4) * 230, 110 + (index // 4) * 150],
            "size": [170, 72],
        }
        if node["symbol"]:
            component["sublabel"] = node["symbol"]
        if repository and node["source_path"] and node["grounding_score"] >= threshold:
            component["sources"] = [_source_evidence(node)]
        components.append(component)

    connections = [
        {
            "id": f"relation-{index:03d}",
            "from": edge["source"],
            "to": edge["target"],
            "label": edge["relation"],
            "route": "auto",
        }
        for index, edge in enumerate(normalized["edges"], start=1)
    ]
    rows = max(1, math.ceil(len(components) / 4))
    meta: dict[str, Any] = {
        "title": document_title,
        "subtitle": f"Source-grounded view of {normalized['artifact_path']}",
        "quality_profile": "standard",
        "viewBox": [1000, max(420, 170 + rows * 150)],
    }
    if repository:
        meta["repository"] = repository
    return {
        "schema_version": ARCHIFY_SCHEMA_VERSION,
        "diagram_type": ARCHIFY_DIAGRAM_TYPE,
        "meta": meta,
        "components": components,
        "connections": connections,
    }


def save_archify_architecture(document: Mapping[str, Any], path: str | Path) -> None:
    """Atomically persist a bounded Archify document produced by this module."""

    if not isinstance(document, Mapping):
        raise ValueError("Archify architecture document must be an object")
    payload = (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > _MAX_OUTPUT_BYTES:
        raise ValueError("Archify architecture document exceeds the byte limit")
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = os.stat(destination, follow_symlinks=False)
    except FileNotFoundError:
        existing_mode = None
    else:
        existing_mode = (
            stat.S_IMODE(existing.st_mode) if stat.S_ISREG(existing.st_mode) else None
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            if existing_mode is not None:
                os.fchmod(stream.fileno(), existing_mode)
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _repository_identity(
    repository_url: str | None,
    revision: str | None,
) -> dict[str, str] | None:
    url = str(repository_url or "").strip()
    commit = str(revision or "").strip()
    if bool(url) != bool(commit):
        raise ValueError("Archify repository URL and revision must be set together")
    if not url:
        return None
    if not _GITHUB_REPOSITORY_RE.fullmatch(url):
        raise ValueError("Archify repository URL must be a public GitHub repository")
    if not _REVISION_RE.fullmatch(commit):
        raise ValueError("Archify revision must be a full 40-character commit SHA")
    return {"url": url, "revision": commit.lower()}


def _source_evidence(node: Mapping[str, Any]) -> dict[str, Any]:
    path = str(node["source_path"])
    if len(path.encode("utf-8")) > _MAX_SOURCE_PATH_BYTES:
        raise ValueError("Archify source path exceeds the schema limit")
    source: dict[str, Any] = {"path": path}
    if node["line"] > 0:
        source["line"] = node["line"]
    if node["symbol"]:
        label = str(node["symbol"])
        if len(label.encode("utf-8")) <= _MAX_SOURCE_LABEL_BYTES:
            source["label"] = label
    return source


def _grounding_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Archify grounding threshold must be numeric")
    threshold = float(value)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Archify grounding threshold must be finite and non-negative")
    return threshold


def _title(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Archify title must be a string")
    title = value.strip()
    if not title or len(title.encode("utf-8")) > _MAX_TITLE_BYTES:
        raise ValueError("Archify title is empty or too long")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in title):
        raise ValueError("Archify title contains control characters")
    return title


def _default_title(artifact_path: str) -> str:
    return f"{Path(artifact_path).stem.replace('-', ' ').replace('_', ' ').title()} architecture"


__all__ = [
    "ARCHIFY_DIAGRAM_TYPE",
    "ARCHIFY_SCHEMA_VERSION",
    "compile_visual_graph_plan_to_archify",
    "save_archify_architecture",
]
