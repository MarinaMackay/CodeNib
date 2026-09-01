# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Compile validated visual graph plans to Archify architecture IR."""

from __future__ import annotations

import io
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .._bounded_json import validate_bounded_json_stream, validate_json_complexity
from ._safe_file_reads import read_regular_bytes
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
_MAX_COMPONENTS = 64
_MAX_CONNECTIONS = 128
_MAX_JSON_NODES = 50_000
_MAX_JSON_TOKENS = 100_000
_MAX_COORDINATE = 100_000.0
_MAX_LINE = 100_000_000
_ID_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]{0,95}")
_DOCUMENT_FIELDS = frozenset(
    {"schema_version", "diagram_type", "meta", "components", "connections"}
)
_META_FIELDS = frozenset(
    {"title", "subtitle", "quality_profile", "viewBox", "repository"}
)
_COMPONENT_FIELDS = frozenset(
    {"id", "type", "label", "sublabel", "pos", "size", "sources"}
)
_CONNECTION_FIELDS = frozenset({"id", "from", "to", "label", "route"})
_SOURCE_FIELDS = frozenset({"path", "line", "label"})
_REPOSITORY_FIELDS = frozenset({"url", "revision"})


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

    validated = validate_archify_architecture(document)
    payload = (
        json.dumps(
            validated,
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


def load_archify_architecture(path: str | Path) -> dict[str, Any]:
    """Load one bounded regular Archify file and validate its complete IR."""

    raw = read_regular_bytes(Path(path).expanduser(), max_bytes=_MAX_OUTPUT_BYTES)
    if raw is None:
        raise ValueError("Archify document must be a stable bounded regular file")
    validate_bounded_json_stream(
        io.BytesIO(raw),
        label="Archify document",
        max_bytes=_MAX_OUTPUT_BYTES,
        max_nodes=_MAX_JSON_NODES,
        max_lexical_tokens=_MAX_JSON_TOKENS,
    )
    try:
        decoded = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_nonfinite_number,
            parse_float=_finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Archify document contains invalid JSON") from exc
    return validate_archify_architecture(decoded)


def validate_archify_architecture(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the safe Archify subset rendered by CodeNib."""

    data = _mapping(document, label="Archify document")
    _exact_fields(data, _DOCUMENT_FIELDS, label="Archify document")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Archify schema version is unsupported")
    if data["diagram_type"] != ARCHIFY_DIAGRAM_TYPE:
        raise ValueError("Archify diagram type is unsupported")

    meta = _validated_meta(data["meta"])
    components = [
        _validated_component(value)
        for value in _mapping_items(
            data["components"], label="Archify components", limit=_MAX_COMPONENTS
        )
    ]
    if not components:
        raise ValueError("Archify document must contain at least one component")
    component_ids = [component["id"] for component in components]
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("Archify component ids must be unique")
    component_id_set = set(component_ids)

    connections = [
        _validated_connection(value)
        for value in _mapping_items(
            data["connections"],
            label="Archify connections",
            limit=_MAX_CONNECTIONS,
        )
    ]
    connection_ids = [connection["id"] for connection in connections]
    if len(connection_ids) != len(set(connection_ids)):
        raise ValueError("Archify connection ids must be unique")
    for connection in connections:
        if (
            connection["from"] not in component_id_set
            or connection["to"] not in component_id_set
        ):
            raise ValueError("Archify connection endpoints must reference components")

    normalized = {
        "schema_version": ARCHIFY_SCHEMA_VERSION,
        "diagram_type": ARCHIFY_DIAGRAM_TYPE,
        "meta": meta,
        "components": components,
        "connections": connections,
    }
    validate_json_complexity(
        normalized, label="Archify document", max_nodes=_MAX_JSON_NODES
    )
    payload = json.dumps(
        normalized, allow_nan=False, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    if len(payload) > _MAX_OUTPUT_BYTES:
        raise ValueError("Archify architecture document exceeds the byte limit")
    return normalized


def _validated_meta(value: Any) -> dict[str, Any]:
    data = _mapping(value, label="Archify meta")
    _exact_fields(data, _META_FIELDS, label="Archify meta", optional={"repository"})
    view_box = _numeric_pair(data["viewBox"], label="Archify viewBox", positive=True)
    normalized: dict[str, Any] = {
        "title": _single_line(data["title"], label="Archify title"),
        "subtitle": _single_line(data["subtitle"], label="Archify subtitle"),
        "quality_profile": _single_line(
            data["quality_profile"], label="Archify quality profile", max_bytes=64
        ),
        "viewBox": view_box,
    }
    if "repository" in data:
        repository = _mapping(data["repository"], label="Archify repository")
        _exact_fields(repository, _REPOSITORY_FIELDS, label="Archify repository")
        normalized["repository"] = _repository_identity(
            repository["url"], repository["revision"]
        )
    return normalized


def _validated_component(value: Any) -> dict[str, Any]:
    data = _mapping(value, label="Archify component")
    _exact_fields(
        data,
        _COMPONENT_FIELDS,
        label="Archify component",
        optional={"sublabel", "sources"},
    )
    normalized: dict[str, Any] = {
        "id": _identifier(data["id"], label="Archify component id"),
        "type": _single_line(
            data["type"], label="Archify component type", max_bytes=64
        ),
        "label": _single_line(data["label"], label="Archify component label"),
        "pos": _numeric_pair(data["pos"], label="Archify component position"),
        "size": _numeric_pair(
            data["size"], label="Archify component size", positive=True
        ),
    }
    if "sublabel" in data:
        normalized["sublabel"] = _single_line(
            data["sublabel"], label="Archify component sublabel"
        )
    if "sources" in data:
        sources = [
            _validated_source(source)
            for source in _mapping_items(
                data["sources"], label="Archify component sources", limit=16
            )
        ]
        source_keys = [(source["path"], source.get("line", 0)) for source in sources]
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("Archify component sources must be unique")
        normalized["sources"] = sources
    return normalized


def _validated_source(value: Any) -> dict[str, Any]:
    data = _mapping(value, label="Archify source")
    _exact_fields(
        data, _SOURCE_FIELDS, label="Archify source", optional={"line", "label"}
    )
    normalized: dict[str, Any] = {
        "path": _relative_path(data["path"], label="Archify source path")
    }
    if "line" in data:
        line = data["line"]
        if type(line) is not int or line < 1 or line > _MAX_LINE:
            raise ValueError("Archify source line is invalid")
        normalized["line"] = line
    if "label" in data:
        normalized["label"] = _single_line(
            data["label"], label="Archify source label", max_bytes=128
        )
    return normalized


def _validated_connection(value: Any) -> dict[str, Any]:
    data = _mapping(value, label="Archify connection")
    _exact_fields(data, _CONNECTION_FIELDS, label="Archify connection")
    route = _single_line(data["route"], label="Archify connection route", max_bytes=32)
    if route != "auto":
        raise ValueError("Archify connection route is unsupported")
    return {
        "id": _identifier(data["id"], label="Archify connection id"),
        "from": _identifier(data["from"], label="Archify connection source"),
        "to": _identifier(data["to"], label="Archify connection target"),
        "label": _single_line(data["label"], label="Archify connection label"),
        "route": route,
    }


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _mapping_items(value: Any, *, label: str, limit: int):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    if len(value) > limit:
        raise ValueError(f"{label} exceeds the item limit")
    for item in value:
        yield _mapping(item, label=f"{label} item")


def _exact_fields(
    data: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    label: str,
    optional: set[str] | frozenset[str] = frozenset(),
) -> None:
    keys = set(data)
    required = set(allowed) - set(optional)
    if keys - set(allowed) or required - keys:
        raise ValueError(f"{label} fields are invalid")


def _single_line(value: Any, *, label: str, max_bytes: int = 512) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    text = value.strip()
    if (
        not text
        or len(text.encode("utf-8")) > max_bytes
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in text)
    ):
        raise ValueError(f"{label} is empty, too long, or contains controls")
    return text


def _identifier(value: Any, *, label: str) -> str:
    text = _single_line(value, label=label, max_bytes=96)
    if not _ID_RE.fullmatch(text):
        raise ValueError(f"{label} is invalid")
    return text


def _numeric_pair(value: Any, *, label: str, positive: bool = False) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain two numbers")
    normalized = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label} must contain finite numbers")
        number = float(item)
        if not math.isfinite(number) or abs(number) > _MAX_COORDINATE:
            raise ValueError(f"{label} must contain bounded finite numbers")
        if positive and number <= 0:
            raise ValueError(f"{label} values must be positive")
        normalized.append(item)
    return normalized


def _relative_path(value: Any, *, label: str) -> str:
    text = _single_line(value, label=label, max_bytes=_MAX_SOURCE_PATH_BYTES)
    if "\\" in text:
        raise ValueError(f"{label} must use POSIX separators")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a safe repository-relative path")
    return path.as_posix()


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Archify document contains duplicate key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_number(value: str) -> float:
    raise ValueError(f"Archify document contains non-finite number: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Archify document contains a non-finite float")
    return parsed


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
    "load_archify_architecture",
    "save_archify_architecture",
    "validate_archify_architecture",
]
