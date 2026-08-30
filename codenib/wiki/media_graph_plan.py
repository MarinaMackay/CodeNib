# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Validated diagram plans derived from source-grounded visual knowledge."""

from __future__ import annotations

import hashlib
import hmac
import html
import io
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .._bounded_json import validate_bounded_json_stream, validate_json_complexity
from ._safe_file_reads import read_regular_bytes
from .media_knowledge import MULTIMODAL_KNOWLEDGE_SCHEMA, MULTIMODAL_KNOWLEDGE_VERSION

VISUAL_GRAPH_PLAN_SCHEMA = "codenib.visual-graph-plan.v1"
VISUAL_GRAPH_PLAN_VERSION = 1
VISUAL_GRAPH_MANIFEST_SCHEMA = "codenib.visual-graph-manifest.v1"
VISUAL_GRAPH_MANIFEST_VERSION = 1

_MAX_PLANS = 4096
_MAX_NODES_PER_PLAN = 24
_MAX_EDGES_PER_PLAN = 48
_MAX_TEXT_BYTES = 4096
_MAX_LINE = 100_000_000
_MAX_SCORE = 1_000_000_000.0
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_MANIFEST_NODES = 1_000_000
_MAX_MANIFEST_TOKENS = 2_000_000
_MAX_KNOWLEDGE_VIEW_BYTES = 128 * 1024 * 1024
_MAX_KNOWLEDGE_VIEW_NODES = 2_000_000
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_NODE_ID_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,95}")
_NODE_ID_CHAR_RE = re.compile(r"[^A-Za-z0-9_]+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PLAN_FIELDS = frozenset(
    {"schema", "version", "artifact_path", "nodes", "edges", "plan_sha256"}
)
_NODE_FIELDS = frozenset({"id", "label", "source_path", "symbol", "line", "evidence"})
_EDGE_FIELDS = frozenset({"source", "target", "relation", "evidence"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "version",
        "knowledge_view_sha256",
        "plan_count",
        "plans",
        "manifest_sha256",
    }
)


def build_visual_graph_plan(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Build one deterministic graph plan from a visual knowledge entry."""

    if not isinstance(entry, Mapping):
        raise ValueError("visual knowledge entry must be an object")
    artifact = _mapping(entry.get("artifact"), label="entry.artifact")
    facts = _mapping(entry.get("facts"), label="entry.facts", allow_empty=True)
    artifact_path = _relative_path(artifact.get("path"), label="artifact_path")
    fact_path = _optional_relative_path(facts.get("artifact_path"), label="fact path")
    if fact_path and fact_path != artifact_path:
        raise ValueError("visual facts are bound to another artifact")

    bindings = list(
        _mapping_items(
            entry.get("bindings") or [],
            label="entry.bindings",
            limit=_MAX_NODES_PER_PLAN * 32,
        )
    )
    entities = list(
        _mapping_items(
            facts.get("entities") or [],
            label="entry.facts.entities",
            limit=_MAX_NODES_PER_PLAN * 4,
        )
    )
    nodes, node_ids = _build_nodes(entities, bindings)
    allowed_sources = {
        path
        for binding in bindings
        if (
            path := _optional_relative_path(
                binding.get("source_path"), label="binding path"
            )
        )
    }
    edges = _build_edges(facts.get("relations") or [], node_ids)
    payload: dict[str, Any] = {
        "schema": VISUAL_GRAPH_PLAN_SCHEMA,
        "version": VISUAL_GRAPH_PLAN_VERSION,
        "artifact_path": artifact_path,
        "nodes": nodes,
        "edges": edges,
    }
    payload["plan_sha256"] = _sha256_json(payload)
    return validate_visual_graph_plan(payload, allowed_source_paths=allowed_sources)


def build_visual_graph_manifest(knowledge_view: Mapping[str, Any]) -> dict[str, Any]:
    """Build a hash-bound graph-plan sidecar for a multimodal knowledge view."""

    view = _validated_knowledge_view(knowledge_view)
    plans = []
    for entry in view["entries"]:
        plan = build_visual_graph_plan(entry)
        if plan["nodes"]:
            plans.append(plan)
    plans.sort(key=lambda plan: plan["artifact_path"])
    payload: dict[str, Any] = {
        "schema": VISUAL_GRAPH_MANIFEST_SCHEMA,
        "version": VISUAL_GRAPH_MANIFEST_VERSION,
        "knowledge_view_sha256": view["view_sha256"],
        "plan_count": len(plans),
        "plans": plans,
    }
    payload["manifest_sha256"] = _sha256_json(payload)
    return validate_visual_graph_manifest(payload)


def validate_visual_graph_plan(
    plan: Mapping[str, Any],
    *,
    allowed_source_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Validate and normalize one persisted visual graph plan."""

    data = _mapping(plan, label="visual graph plan")
    _exact_fields(data, _PLAN_FIELDS, label="visual graph plan")
    if data["schema"] != VISUAL_GRAPH_PLAN_SCHEMA:
        raise ValueError("visual graph plan schema is unsupported")
    if type(data["version"]) is not int or data["version"] != VISUAL_GRAPH_PLAN_VERSION:
        raise ValueError("visual graph plan version is unsupported")
    artifact_path = _relative_path(data["artifact_path"], label="artifact_path")
    nodes = [
        _validated_node(value)
        for value in _mapping_items(
            data["nodes"], label="visual graph plan nodes", limit=_MAX_NODES_PER_PLAN
        )
    ]
    edges = [
        _validated_edge(value)
        for value in _mapping_items(
            data["edges"], label="visual graph plan edges", limit=_MAX_EDGES_PER_PLAN
        )
    ]
    if not nodes:
        raise ValueError("visual graph plan must contain at least one node")
    node_ids = [node["id"] for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("visual graph node ids must be unique")
    node_id_set = set(node_ids)
    normalized_allowed = (
        {
            _relative_path(path, label="allowed source path")
            for path in allowed_source_paths
        }
        if allowed_source_paths is not None
        else None
    )
    for node in nodes:
        source_path = node["source_path"]
        if (
            source_path
            and normalized_allowed is not None
            and source_path not in normalized_allowed
        ):
            raise ValueError("visual graph node source_path is not allowed")
    edge_keys = []
    for edge in edges:
        if edge["source"] not in node_id_set or edge["target"] not in node_id_set:
            raise ValueError("visual graph edge endpoints must reference nodes")
        edge_keys.append((edge["source"], edge["target"], edge["relation"]))
    if len(edge_keys) != len(set(edge_keys)):
        raise ValueError("visual graph edges must be unique")

    normalized: dict[str, Any] = {
        "schema": VISUAL_GRAPH_PLAN_SCHEMA,
        "version": VISUAL_GRAPH_PLAN_VERSION,
        "artifact_path": artifact_path,
        "nodes": nodes,
        "edges": edges,
    }
    expected = _sha256_json(normalized)
    recorded = _digest(data["plan_sha256"], label="plan_sha256")
    if not hmac.compare_digest(recorded, expected):
        raise ValueError("visual graph plan hash does not match")
    normalized["plan_sha256"] = recorded
    return normalized


def validate_visual_graph_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a persisted visual graph manifest."""

    data = _mapping(manifest, label="visual graph manifest")
    _exact_fields(data, _MANIFEST_FIELDS, label="visual graph manifest")
    if data["schema"] != VISUAL_GRAPH_MANIFEST_SCHEMA:
        raise ValueError("visual graph manifest schema is unsupported")
    if (
        type(data["version"]) is not int
        or data["version"] != VISUAL_GRAPH_MANIFEST_VERSION
    ):
        raise ValueError("visual graph manifest version is unsupported")
    plans = [
        validate_visual_graph_plan(value)
        for value in _mapping_items(
            data["plans"], label="visual graph manifest plans", limit=_MAX_PLANS
        )
    ]
    if type(data["plan_count"]) is not int or data["plan_count"] != len(plans):
        raise ValueError("visual graph manifest plan_count is invalid")
    paths = [plan["artifact_path"] for plan in plans]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("visual graph manifest plans must have unique sorted paths")
    view_sha256 = _digest(data["knowledge_view_sha256"], label="knowledge_view_sha256")
    normalized: dict[str, Any] = {
        "schema": VISUAL_GRAPH_MANIFEST_SCHEMA,
        "version": VISUAL_GRAPH_MANIFEST_VERSION,
        "knowledge_view_sha256": view_sha256,
        "plan_count": len(plans),
        "plans": plans,
    }
    expected = _sha256_json(normalized)
    recorded = _digest(data["manifest_sha256"], label="manifest_sha256")
    if not hmac.compare_digest(recorded, expected):
        raise ValueError("visual graph manifest hash does not match")
    normalized["manifest_sha256"] = recorded
    validate_json_complexity(
        normalized,
        label="visual graph manifest",
        max_nodes=_MAX_MANIFEST_NODES,
    )
    if len(_canonical_json_bytes(normalized)) > _MAX_MANIFEST_BYTES:
        raise ValueError("visual graph manifest exceeds the byte limit")
    return normalized


def compile_visual_graph_plan_to_mermaid(plan: Mapping[str, Any]) -> str:
    """Compile a validated graph plan to conservative Mermaid flowchart text."""

    normalized = validate_visual_graph_plan(plan)
    lines = ["flowchart LR"]
    for node in normalized["nodes"]:
        lines.append(f'  {node["id"]}["{_mermaid_text(node["label"])}"]')
    for edge in normalized["edges"]:
        relation = _mermaid_text(edge["relation"], edge_label=True)
        lines.append(f'  {edge["source"]} -->|{relation}| {edge["target"]}')
    return "\n".join(lines) + "\n"


def save_visual_graph_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    """Atomically persist a validated visual graph manifest."""

    validated = validate_visual_graph_manifest(manifest)
    payload = (
        json.dumps(
            validated, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise ValueError("visual graph manifest exceeds the byte limit")
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


def load_visual_graph_manifest(path: str | Path) -> dict[str, Any]:
    """Load a bounded regular JSON file and validate its graph manifest."""

    raw = read_regular_bytes(Path(path).expanduser(), max_bytes=_MAX_MANIFEST_BYTES)
    if raw is None:
        raise ValueError("visual graph manifest must be a stable bounded regular file")
    validate_bounded_json_stream(
        io.BytesIO(raw),
        label="visual graph manifest",
        max_bytes=_MAX_MANIFEST_BYTES,
        max_nodes=_MAX_MANIFEST_NODES,
        max_lexical_tokens=_MAX_MANIFEST_TOKENS,
    )
    try:
        decoded = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_nonfinite_number,
            parse_float=_finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("visual graph manifest contains invalid JSON") from exc
    return validate_visual_graph_manifest(decoded)


def _build_nodes(
    entities: list[Mapping[str, Any]], bindings: list[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    ordered_labels: list[str] = []
    entity_by_label: dict[str, Mapping[str, Any]] = {}
    for entity in entities:
        label = _required_single_line(entity.get("name"), label="entity name")
        if label not in entity_by_label:
            ordered_labels.append(label)
            entity_by_label[label] = entity
    bindings_by_label: dict[str, list[Mapping[str, Any]]] = {}
    for binding in bindings:
        label = _required_single_line(
            binding.get("entity_name"), label="binding entity"
        )
        bindings_by_label.setdefault(label, []).append(binding)
        if label not in entity_by_label and label not in ordered_labels:
            ordered_labels.append(label)
    ordered_labels = ordered_labels[:_MAX_NODES_PER_PLAN]
    used_ids: set[str] = set()
    node_ids: dict[str, str] = {}
    nodes = []
    for label in ordered_labels:
        binding = _best_binding(bindings_by_label.get(label, ()))
        entity = entity_by_label.get(label, {})
        node_id = _unique_node_id(label, used_ids)
        used_ids.add(node_id)
        node_ids[label] = node_id
        source_path = (
            _optional_relative_path(binding.get("source_path"), label="binding path")
            if binding
            else ""
        )
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "source_path": source_path,
                "symbol": _optional_single_line(
                    binding.get("symbol") if binding else "", label="binding symbol"
                ),
                "line": _line(binding.get("line") if binding else 0),
                "evidence": _text(
                    entity.get("evidence")
                    or (binding.get("evidence") if binding else ""),
                    label="node evidence",
                ),
            }
        )
    return nodes, node_ids


def _build_edges(value: Any, node_ids: Mapping[str, str]) -> list[dict[str, Any]]:
    edges = []
    seen = set()
    for relation in _mapping_items(
        value, label="entry.facts.relations", limit=_MAX_EDGES_PER_PLAN * 4
    ):
        source_label = _required_single_line(
            relation.get("source"), label="relation source"
        )
        target_label = _required_single_line(
            relation.get("target"), label="relation target"
        )
        if source_label not in node_ids or target_label not in node_ids:
            continue
        edge = {
            "source": node_ids[source_label],
            "target": node_ids[target_label],
            "relation": _required_single_line(
                relation.get("relation") or "related_to", label="relation"
            ),
            "evidence": _text(relation.get("evidence"), label="edge evidence"),
        }
        key = (edge["source"], edge["target"], edge["relation"])
        if key not in seen:
            seen.add(key)
            edges.append(edge)
        if len(edges) >= _MAX_EDGES_PER_PLAN:
            break
    return edges


def _best_binding(bindings: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    ranked = []
    for binding in bindings:
        source_path = _optional_relative_path(
            binding.get("source_path"), label="binding path"
        )
        if not source_path:
            continue
        ranked.append(
            (
                -_score(binding.get("score")),
                source_path,
                _optional_single_line(binding.get("symbol"), label="binding symbol"),
                _line(binding.get("line")),
                binding,
            )
        )
    ranked.sort(key=lambda item: item[:-1])
    return ranked[0][-1] if ranked else None


def _validated_node(value: Any) -> dict[str, Any]:
    node = _mapping(value, label="visual graph node")
    _exact_fields(node, _NODE_FIELDS, label="visual graph node")
    node_id = _required_single_line(node["id"], label="node id")
    if _NODE_ID_RE.fullmatch(node_id) is None:
        raise ValueError("visual graph node id is invalid")
    return {
        "id": node_id,
        "label": _required_single_line(node["label"], label="node label"),
        "source_path": _optional_relative_path(
            node["source_path"], label="source_path"
        ),
        "symbol": _optional_single_line(node["symbol"], label="node symbol"),
        "line": _line(node["line"]),
        "evidence": _text(node["evidence"], label="node evidence"),
    }


def _validated_edge(value: Any) -> dict[str, Any]:
    edge = _mapping(value, label="visual graph edge")
    _exact_fields(edge, _EDGE_FIELDS, label="visual graph edge")
    source = _required_single_line(edge["source"], label="edge source")
    target = _required_single_line(edge["target"], label="edge target")
    if _NODE_ID_RE.fullmatch(source) is None or _NODE_ID_RE.fullmatch(target) is None:
        raise ValueError("visual graph edge endpoint is invalid")
    return {
        "source": source,
        "target": target,
        "relation": _required_single_line(edge["relation"], label="edge relation"),
        "evidence": _text(edge["evidence"], label="edge evidence"),
    }


def _validated_knowledge_view(value: Mapping[str, Any]) -> dict[str, Any]:
    view = _mapping(value, label="multimodal knowledge view")
    validate_json_complexity(
        view,
        label="multimodal knowledge view",
        max_nodes=_MAX_KNOWLEDGE_VIEW_NODES,
    )
    if len(_canonical_json_bytes(view)) > _MAX_KNOWLEDGE_VIEW_BYTES:
        raise ValueError("multimodal knowledge view exceeds the byte limit")
    if view.get("schema") != MULTIMODAL_KNOWLEDGE_SCHEMA:
        raise ValueError("multimodal knowledge view schema is unsupported")
    if (
        type(view.get("version")) is not int
        or view["version"] != MULTIMODAL_KNOWLEDGE_VERSION
    ):
        raise ValueError("multimodal knowledge view version is unsupported")
    entries = list(
        _mapping_items(
            view.get("entries"), label="knowledge view entries", limit=_MAX_PLANS
        )
    )
    if type(view.get("entry_count")) is not int or view["entry_count"] != len(entries):
        raise ValueError("multimodal knowledge view entry_count is invalid")
    recorded = _digest(view.get("view_sha256"), label="view_sha256")
    expected = _sha256_json(
        {key: item for key, item in view.items() if key != "view_sha256"}
    )
    if not hmac.compare_digest(recorded, expected):
        raise ValueError("multimodal knowledge view hash does not match")
    return {**dict(view), "entries": entries, "view_sha256": recorded}


def _mapping(value: Any, *, label: str, allow_empty: bool = False) -> dict[str, Any]:
    if value is None and allow_empty:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _mapping_items(
    value: Any, *, label: str, limit: int
) -> Iterable[Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"{label} must be a bounded array")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{label} must contain only objects")
    return value


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields are invalid")


def _relative_path(value: Any, *, label: str) -> str:
    text = _required_single_line(value, label=label)
    path = PurePosixPath(text)
    if (
        text == "."
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != text
        or "\\" in text
    ):
        raise ValueError(f"visual graph {label} must be repository-relative")
    return text


def _optional_relative_path(value: Any, *, label: str) -> str:
    text = _optional_single_line(value, label=label)
    return _relative_path(text, label=label) if text else ""


def _required_single_line(value: Any, *, label: str) -> str:
    text = _optional_single_line(value, label=label)
    if not text:
        raise ValueError(f"visual graph {label} is required")
    return text


def _optional_single_line(value: Any, *, label: str) -> str:
    text = _text(value, label=label)
    if "\n" in text or "\r" in text:
        raise ValueError(f"visual graph {label} must be one line")
    return text


def _text(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"visual graph {label} must be text")
    text = value.strip()
    if _CONTROL_RE.search(text):
        raise ValueError(f"visual graph {label} contains control characters")
    if len(text.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise ValueError(f"visual graph {label} exceeds the byte limit")
    return text


def _line(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_LINE:
        raise ValueError("visual graph line is invalid")
    return value


def _score(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score) or not 0 <= score <= _MAX_SCORE:
        return 0.0
    return score


def _unique_node_id(label: str, used: set[str]) -> str:
    stem = _NODE_ID_CHAR_RE.sub("_", label).strip("_") or "node"
    if stem[0].isdigit():
        stem = f"n_{stem}"
    stem = stem[:96]
    if stem not in used:
        return stem
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:10]
    candidate = f"{stem[:85]}_{digest}"
    counter = 2
    while candidate in used:
        suffix = f"_{counter}"
        candidate = f"{stem[:96-len(suffix)]}{suffix}"
        counter += 1
    return candidate


def _mermaid_text(value: str, *, edge_label: bool = False) -> str:
    escaped = (
        html.escape(value, quote=True)
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace("{", "&#123;")
        .replace("}", "&#125;")
        .replace("`", "&#96;")
    )
    if edge_label:
        escaped = escaped.replace("|", "&#124;")
    return escaped


def _digest(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"visual graph {label} is invalid")
    return value


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValueError("visual graph payload must contain bounded JSON") from exc


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_number(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number")
    return result


__all__ = [
    "VISUAL_GRAPH_MANIFEST_SCHEMA",
    "VISUAL_GRAPH_MANIFEST_VERSION",
    "VISUAL_GRAPH_PLAN_SCHEMA",
    "VISUAL_GRAPH_PLAN_VERSION",
    "build_visual_graph_manifest",
    "build_visual_graph_plan",
    "compile_visual_graph_plan_to_mermaid",
    "load_visual_graph_manifest",
    "save_visual_graph_manifest",
    "validate_visual_graph_manifest",
    "validate_visual_graph_plan",
]
