# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded reader-facing summaries for multimodal repository bundles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

_SUMMARY_LIMITS = {
    "artifacts": 8,
    "facts": 8,
    "bindings": 16,
    "graph_plans": 8,
    "storyboards": 8,
    "entities": 8,
    "claims": 3,
    "relations": 8,
    "graph_nodes": 12,
    "graph_edges": 16,
    "storyboard_frames": 5,
    "storyboard_focus_nodes": 8,
    "storyboard_citations": 8,
    "incremental_paths": 16,
}
_TEXT_BYTES = 512
_LONG_TEXT_BYTES = 2048
_PATH_BYTES = 4096


def summarize_multimodal_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Return a bounded, redacted summary suitable for the local Wiki UI."""

    media_manifest = bundle.get("media_manifest") or {}
    visual_facts_manifest = bundle.get("visual_facts_manifest") or {}
    grounding_manifest = bundle.get("grounding_manifest") or {}
    knowledge_view = bundle.get("knowledge_view") or {}
    visual_graph_manifest = bundle.get("visual_graph_manifest") or {}
    visual_storyboard_manifest = bundle.get("visual_storyboard_manifest") or {}
    return {
        "schema": bundle.get("schema"),
        "schema_version": bundle.get("schema_version"),
        "bundle_sha256": bundle.get("bundle_sha256"),
        "source_candidate_count": bundle.get("source_candidate_count", 0),
        "media_manifest": {
            "artifact_count": media_manifest.get("artifact_count", 0),
            "artifacts": _items(
                media_manifest.get("artifacts"),
                _artifact,
                limit=_SUMMARY_LIMITS["artifacts"],
            ),
        },
        "visual_facts_manifest": {
            "fact_count": visual_facts_manifest.get("fact_count", 0),
            "facts": _items(
                visual_facts_manifest.get("facts"),
                _fact_pack,
                limit=_SUMMARY_LIMITS["facts"],
            ),
        },
        "grounding_manifest": {
            "binding_count": grounding_manifest.get("binding_count", 0),
            "bindings": _items(
                grounding_manifest.get("bindings"),
                _binding,
                limit=_SUMMARY_LIMITS["bindings"],
            ),
        },
        "knowledge_view": {
            "entry_count": knowledge_view.get("entry_count", 0),
        },
        "visual_graph_manifest": {
            "plan_count": visual_graph_manifest.get("plan_count", 0),
            "plans": _items(
                visual_graph_manifest.get("plans"),
                _graph_plan,
                limit=_SUMMARY_LIMITS["graph_plans"],
            ),
        },
        "visual_storyboard_manifest": {
            "storyboard_count": visual_storyboard_manifest.get("storyboard_count", 0),
            "storyboards": _items(
                visual_storyboard_manifest.get("storyboards"),
                _storyboard,
                limit=_SUMMARY_LIMITS["storyboards"],
            ),
        },
        "incremental_update": _incremental_update(
            bundle.get("incremental_update") or {}
        ),
    }


def _items(values, summarizer, *, limit: int) -> list[dict[str, Any]]:
    summarized = []
    for value in list(values or [])[: max(0, limit)]:
        if not isinstance(value, Mapping):
            continue
        item = summarizer(value)
        if item:
            summarized.append(item)
    return summarized


def _artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    path = _path(artifact.get("path"))
    if not path:
        return {}
    return _compact(
        {
            "path": path,
            "media_type": _text(artifact.get("media_type")),
            "mime_type": _text(artifact.get("mime_type")),
            "role_hint": _text(artifact.get("role_hint")),
            "caption": _text(artifact.get("caption")),
            "sha256": _hash(artifact.get("sha256")),
            "size_bytes": _int(artifact.get("size_bytes")),
        }
    )


def _fact_pack(fact: Mapping[str, Any]) -> dict[str, Any]:
    artifact_path = _path(fact.get("artifact_path"))
    if not artifact_path:
        return {}
    return _compact(
        {
            "artifact_path": artifact_path,
            "extractor": _text(fact.get("extractor")),
            "role_hint": _text(fact.get("role_hint")),
            "entities": _items(
                fact.get("entities"),
                _entity,
                limit=_SUMMARY_LIMITS["entities"],
            ),
            "claims": _items(
                fact.get("claims"),
                _claim,
                limit=_SUMMARY_LIMITS["claims"],
            ),
            "relations": _items(
                fact.get("relations"),
                _relation,
                limit=_SUMMARY_LIMITS["relations"],
            ),
        }
    )


def _entity(entity: Mapping[str, Any]) -> dict[str, Any]:
    return _compact(
        {
            "name": _text(entity.get("name")),
            "type": _text(entity.get("type")),
            "evidence": _text(entity.get("evidence")),
            "confidence": _score(entity.get("confidence")),
            "grounding_candidates": [
                candidate
                for candidate in (
                    _text(value)
                    for value in list(entity.get("grounding_candidates") or [])[:8]
                )
                if candidate
            ],
        }
    )


def _claim(claim: Mapping[str, Any]) -> dict[str, Any]:
    return _compact(
        {
            "text": _text(claim.get("text"), max_bytes=_LONG_TEXT_BYTES),
            "evidence": _text(claim.get("evidence")),
            "confidence": _score(claim.get("confidence")),
        }
    )


def _relation(relation: Mapping[str, Any]) -> dict[str, Any]:
    return _compact(
        {
            "source": _text(relation.get("source")),
            "target": _text(relation.get("target")),
            "relation": _text(relation.get("relation")),
            "evidence": _text(relation.get("evidence")),
            "confidence": _score(relation.get("confidence")),
        }
    )


def _binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    source_path = _path(binding.get("source_path"))
    if not source_path:
        return {}
    return _compact(
        {
            "artifact_path": _path(binding.get("artifact_path")),
            "entity_name": _text(binding.get("entity_name")),
            "source_path": source_path,
            "symbol": _text(binding.get("symbol")),
            "kind": _text(binding.get("kind")),
            "line": _int(binding.get("line")),
            "score": _score(binding.get("score")),
            "evidence": _text(binding.get("evidence")),
        }
    )


def _graph_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    artifact_path = _path(plan.get("artifact_path"))
    if not artifact_path:
        return {}
    return _compact(
        {
            "artifact_path": artifact_path,
            "nodes": _items(
                plan.get("nodes"),
                _graph_node,
                limit=_SUMMARY_LIMITS["graph_nodes"],
            ),
            "edges": _items(
                plan.get("edges"),
                _graph_edge,
                limit=_SUMMARY_LIMITS["graph_edges"],
            ),
        }
    )


def _graph_node(node: Mapping[str, Any]) -> dict[str, Any]:
    return _compact(
        {
            "id": _text(node.get("id")),
            "label": _text(node.get("label")),
            "source_path": _path(node.get("source_path")),
            "symbol": _text(node.get("symbol")),
            "line": _int(node.get("line")),
            "evidence": _text(node.get("evidence")),
        }
    )


def _graph_edge(edge: Mapping[str, Any]) -> dict[str, Any]:
    return _compact(
        {
            "source": _text(edge.get("source")),
            "target": _text(edge.get("target")),
            "relation": _text(edge.get("relation")),
            "evidence": _text(edge.get("evidence")),
        }
    )


def _storyboard(storyboard: Mapping[str, Any]) -> dict[str, Any]:
    artifact_path = _path(storyboard.get("artifact_path"))
    if not artifact_path:
        return {}
    return _compact(
        {
            "artifact_path": artifact_path,
            "title": _text(storyboard.get("title")),
            "frames": _items(
                storyboard.get("frames"),
                _storyboard_frame,
                limit=_SUMMARY_LIMITS["storyboard_frames"],
            ),
        }
    )


def _storyboard_frame(frame: Mapping[str, Any]) -> dict[str, Any]:
    return _compact(
        {
            "id": _text(frame.get("id")),
            "title": _text(frame.get("title")),
            "narration": _text(
                frame.get("narration"), max_bytes=_LONG_TEXT_BYTES
            ),
            "visual_prompt": _text(
                frame.get("visual_prompt"),
                max_bytes=_LONG_TEXT_BYTES,
            ),
            "duration_ms": _int(frame.get("duration_ms")),
            "focus_node_ids": [
                node_id
                for node_id in (
                    _text(value)
                    for value in list(frame.get("focus_node_ids") or [])[
                        : _SUMMARY_LIMITS["storyboard_focus_nodes"]
                    ]
                )
                if node_id
            ],
            "source_citations": [
                path
                for path in (
                    _path(value)
                    for value in list(frame.get("source_citations") or [])[
                        : _SUMMARY_LIMITS["storyboard_citations"]
                    ]
                )
                if path
            ],
        }
    )


def _incremental_update(update: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(update, Mapping):
        return {}
    media_diff = update.get("media_diff") or {}
    counts = media_diff.get("counts") if isinstance(media_diff, Mapping) else {}
    return _compact(
        {
            "counts": {
                key: _int(value)
                for key, value in dict(counts or {}).items()
                if key in {"added", "removed", "changed", "unchanged"}
            },
            "extract_artifact_paths": [
                path
                for path in (
                    _path(value)
                    for value in list(update.get("extract_artifact_paths") or [])[
                        : _SUMMARY_LIMITS["incremental_paths"]
                    ]
                )
                if path
            ],
            "removed_artifact_paths": [
                path
                for path in (
                    _path(value)
                    for value in list(update.get("removed_artifact_paths") or [])[
                        : _SUMMARY_LIMITS["incremental_paths"]
                    ]
                )
                if path
            ],
        }
    )


def _path(value: Any) -> str:
    text = _text(value, max_bytes=_PATH_BYTES)
    if not text or "\\" in text:
        return ""
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return ""
    return path.as_posix()


def _text(value: Any, *, max_bytes: int = _TEXT_BYTES) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = "".join(
        character
        for character in text
        if ord(character) >= 0x20 and ord(character) != 0x7F
    )
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", "ignore")


def _hash(value: Any) -> str:
    text = str(value or "").strip().lower()
    if len(text) == 64 and all(character in "0123456789abcdef" for character in text):
        return text
    return ""


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= score <= 1.0:
        return None
    return round(score, 4)


def _compact(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if item not in ("", None, [], {})
    }


__all__ = ["summarize_multimodal_bundle"]
