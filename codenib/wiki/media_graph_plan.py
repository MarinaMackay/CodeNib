# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Validated visual graph plans derived from multimodal knowledge views."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping

VISUAL_GRAPH_PLAN_SCHEMA = "codenib.visual-graph-plan.v1"
VISUAL_GRAPH_PLAN_VERSION = 1
VISUAL_GRAPH_MANIFEST_SCHEMA = "codenib.visual-graph-manifest.v1"
VISUAL_GRAPH_MANIFEST_VERSION = 1

_MAX_NODES_PER_PLAN = 24
_MAX_EDGES_PER_PLAN = 48
_MAX_TEXT_BYTES = 4096
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MERMAID_ID_RE = re.compile(r"[^A-Za-z0-9_]")


@dataclass(frozen=True)
class VisualGraphNode:
    id: str
    label: str
    source_path: str = ""
    symbol: str = ""
    line: int = 0
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualGraphEdge:
    source: str
    target: str
    relation: str = "related_to"
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualGraphPlan:
    artifact_path: str
    nodes: tuple[VisualGraphNode, ...]
    edges: tuple[VisualGraphEdge, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": VISUAL_GRAPH_PLAN_SCHEMA,
            "version": VISUAL_GRAPH_PLAN_VERSION,
            "artifact_path": self.artifact_path,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }
        payload["plan_sha256"] = _sha256_json(
            {key: value for key, value in payload.items() if key != "plan_sha256"}
        )
        return payload


def build_visual_graph_manifest(
    knowledge_view: Mapping[str, Any],
) -> dict[str, Any]:
    """Build validated visual graph plans for each multimodal knowledge entry."""

    plans = []
    for entry in knowledge_view.get("entries") or ():
        if not isinstance(entry, Mapping):
            continue
        plan = build_visual_graph_plan(entry)
        if plan["nodes"]:
            validate_visual_graph_plan(plan)
            plans.append(plan)
    payload = {
        "schema": VISUAL_GRAPH_MANIFEST_SCHEMA,
        "version": VISUAL_GRAPH_MANIFEST_VERSION,
        "knowledge_view_sha256": str(knowledge_view.get("view_sha256") or ""),
        "plan_count": len(plans),
        "plans": sorted(plans, key=lambda plan: plan["artifact_path"]),
    }
    payload["manifest_sha256"] = _sha256_json(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    return payload


def build_visual_graph_plan(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Build one validated, renderable graph plan from a knowledge entry."""

    artifact = entry.get("artifact") or {}
    fact = entry.get("facts") or {}
    artifact_path = _safe_text((artifact or {}).get("path"))
    nodes = _graph_nodes(entry)
    edges = _graph_edges(
        nodes,
        artifact=artifact if isinstance(artifact, Mapping) else {},
        fact=fact if isinstance(fact, Mapping) else {},
    )
    plan = VisualGraphPlan(
        artifact_path=artifact_path,
        nodes=tuple(nodes[:_MAX_NODES_PER_PLAN]),
        edges=tuple(edges[:_MAX_EDGES_PER_PLAN]),
    ).to_dict()
    validate_visual_graph_plan(plan)
    return plan


def validate_visual_graph_plan(
    plan: Mapping[str, Any],
    *,
    allowed_source_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Validate a visual graph plan and return a normalized dictionary."""

    if not isinstance(plan, Mapping):
        raise ValueError("visual graph plan must be an object")
    if plan.get("schema") != VISUAL_GRAPH_PLAN_SCHEMA:
        raise ValueError("visual graph plan schema is unsupported")
    if plan.get("version") != VISUAL_GRAPH_PLAN_VERSION:
        raise ValueError("visual graph plan version is unsupported")
    artifact_path = _safe_relative_path(
        plan.get("artifact_path"), label="artifact_path"
    )
    nodes = [_node_from_mapping(node) for node in plan.get("nodes") or ()]
    edges = [_edge_from_mapping(edge) for edge in plan.get("edges") or ()]
    if len(nodes) > _MAX_NODES_PER_PLAN:
        raise ValueError("visual graph plan has too many nodes")
    if len(edges) > _MAX_EDGES_PER_PLAN:
        raise ValueError("visual graph plan has too many edges")
    ids = [node.id for node in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("visual graph node ids must be unique")
    id_set = set(ids)
    for node in nodes:
        if node.source_path:
            _safe_relative_path(node.source_path, label="source_path")
            if (
                allowed_source_paths is not None
                and node.source_path not in allowed_source_paths
            ):
                raise ValueError("visual graph node source_path is not allowed")
    for edge in edges:
        if edge.source not in id_set or edge.target not in id_set:
            raise ValueError("visual graph edge endpoints must reference nodes")
    normalized = VisualGraphPlan(
        artifact_path=artifact_path,
        nodes=tuple(nodes),
        edges=tuple(edges),
    ).to_dict()
    recorded = str(plan.get("plan_sha256") or "")
    if recorded and recorded != normalized["plan_sha256"]:
        raise ValueError("visual graph plan hash does not match")
    return normalized


def compile_visual_graph_plan_to_mermaid(plan: Mapping[str, Any]) -> str:
    """Compile a validated visual graph plan to Mermaid flowchart syntax."""

    normalized = validate_visual_graph_plan(plan)
    lines = ["flowchart LR"]
    id_map = {}
    for index, node in enumerate(normalized["nodes"], start=1):
        mermaid_id = _mermaid_id(node["id"], index)
        id_map[node["id"]] = mermaid_id
        lines.append(f'  {mermaid_id}["{_escape_mermaid(node["label"])}"]')
    for edge in normalized["edges"]:
        source = id_map[edge["source"]]
        target = id_map[edge["target"]]
        relation = _escape_mermaid(edge.get("relation") or "related_to")
        lines.append(f"  {source} -->|{relation}| {target}")
    return "\n".join(lines) + "\n"


def _graph_nodes(entry: Mapping[str, Any]) -> list[VisualGraphNode]:
    by_id: dict[str, VisualGraphNode] = {}
    bindings = [
        binding
        for binding in entry.get("bindings") or ()
        if isinstance(binding, Mapping)
    ]
    for binding in bindings:
        label = _safe_text(binding.get("entity_name"))
        if not label:
            continue
        node_id = _node_id(label)
        by_id.setdefault(
            node_id,
            VisualGraphNode(
                id=node_id,
                label=label,
                source_path=_safe_text(binding.get("source_path")),
                symbol=_safe_text(binding.get("symbol")),
                line=_int(binding.get("line")),
                evidence=_safe_text(binding.get("evidence")),
            ),
        )
    facts = entry.get("facts") or {}
    if isinstance(facts, Mapping):
        for entity in facts.get("entities") or ():
            if not isinstance(entity, Mapping):
                continue
            label = _safe_text(entity.get("name"))
            if not label:
                continue
            by_id.setdefault(
                _node_id(label),
                VisualGraphNode(
                    id=_node_id(label),
                    label=label,
                    evidence=_safe_text(entity.get("evidence")),
                ),
            )
    return sorted(by_id.values(), key=lambda node: (node.label.lower(), node.id))


def _graph_edges(
    nodes: list[VisualGraphNode],
    *,
    artifact: Mapping[str, Any],
    fact: Mapping[str, Any],
) -> list[VisualGraphEdge]:
    by_label = {node.label: node for node in nodes}
    edges: dict[tuple[str, str, str], VisualGraphEdge] = {}
    for relation in fact.get("relations") or ():
        if not isinstance(relation, Mapping):
            continue
        source = by_label.get(_safe_text(relation.get("source")))
        target = by_label.get(_safe_text(relation.get("target")))
        if source is None or target is None or source.id == target.id:
            continue
        edge = VisualGraphEdge(
            source=source.id,
            target=target.id,
            relation=_safe_text(relation.get("relation") or "related_to"),
            evidence=_safe_text(relation.get("evidence")),
        )
        edges[(edge.source, edge.target, edge.relation)] = edge
    text = " ".join(
        [
            _safe_text(artifact.get("embedded_text")),
            _safe_text(artifact.get("caption")),
            *[
                _safe_text(claim.get("text")) + " " + _safe_text(claim.get("evidence"))
                for claim in fact.get("claims") or ()
                if isinstance(claim, Mapping)
            ],
        ]
    )
    labels = list(by_label)
    for source_label in labels:
        for target_label in labels:
            if source_label == target_label:
                continue
            if _mentions_call(text, source_label, target_label):
                source = by_label[source_label]
                target = by_label[target_label]
                edge = VisualGraphEdge(
                    source=source.id,
                    target=target.id,
                    relation="calls",
                    evidence=_safe_text(text, max_bytes=512),
                )
                edges.setdefault((edge.source, edge.target, edge.relation), edge)
    if not edges and len(nodes) >= 2:
        ordered = sorted(
            nodes, key=lambda node: (node.line <= 0, node.line, node.label)
        )
        for source, target in zip(ordered, ordered[1:]):
            edge = VisualGraphEdge(
                source=source.id,
                target=target.id,
                relation="co_occurs_with",
                evidence=_safe_text(artifact.get("path")),
            )
            edges.setdefault((edge.source, edge.target, edge.relation), edge)
    return sorted(
        edges.values(), key=lambda edge: (edge.source, edge.target, edge.relation)
    )


def _mentions_call(text: str, source: str, target: str) -> bool:
    lowered = text.lower()
    source_index = lowered.find(source.lower())
    target_index = lowered.find(target.lower())
    if source_index < 0 or target_index < 0 or source_index >= target_index:
        return False
    window = lowered[source_index : target_index + len(target)]
    return "call" in window or "->" in window or "→" in window


def _node_from_mapping(value: Any) -> VisualGraphNode:
    if not isinstance(value, Mapping):
        raise ValueError("visual graph node must be an object")
    node_id = _safe_text(value.get("id"))
    label = _safe_text(value.get("label"))
    if not node_id or not label:
        raise ValueError("visual graph node id and label are required")
    return VisualGraphNode(
        id=node_id,
        label=label,
        source_path=_safe_text(value.get("source_path")),
        symbol=_safe_text(value.get("symbol")),
        line=_int(value.get("line")),
        evidence=_safe_text(value.get("evidence")),
    )


def _edge_from_mapping(value: Any) -> VisualGraphEdge:
    if not isinstance(value, Mapping):
        raise ValueError("visual graph edge must be an object")
    source = _safe_text(value.get("source"))
    target = _safe_text(value.get("target"))
    if not source or not target:
        raise ValueError("visual graph edge source and target are required")
    return VisualGraphEdge(
        source=source,
        target=target,
        relation=_safe_text(value.get("relation") or "related_to"),
        evidence=_safe_text(value.get("evidence")),
    )


def _node_id(label: str) -> str:
    value = _MERMAID_ID_RE.sub("_", label.strip())
    value = value.strip("_") or "node"
    if value[0].isdigit():
        value = f"n_{value}"
    return value[:96]


def _mermaid_id(node_id: str, index: int) -> str:
    value = _node_id(node_id)
    return value or f"node_{index}"


def _safe_relative_path(value: Any, *, label: str) -> str:
    text = _safe_text(value)
    if not text:
        raise ValueError(f"visual graph {label} is required")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise ValueError(f"visual graph {label} must be repository-relative")
    return path.as_posix()


def _safe_text(value: Any, *, max_bytes: int = _MAX_TEXT_BYTES) -> str:
    text = str(value or "").strip()
    if _CONTROL_RE.search(text):
        raise ValueError("visual graph text contains control characters")
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    encoded = text.encode("utf-8")[: max(0, max_bytes - 1)]
    return encoded.decode("utf-8", errors="ignore").rstrip() + "…"


def _escape_mermaid(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "VISUAL_GRAPH_MANIFEST_SCHEMA",
    "VISUAL_GRAPH_MANIFEST_VERSION",
    "VISUAL_GRAPH_PLAN_SCHEMA",
    "VISUAL_GRAPH_PLAN_VERSION",
    "VisualGraphEdge",
    "VisualGraphNode",
    "VisualGraphPlan",
    "build_visual_graph_manifest",
    "build_visual_graph_plan",
    "compile_visual_graph_plan_to_mermaid",
    "validate_visual_graph_plan",
]
