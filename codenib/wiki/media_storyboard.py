# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Video-ready storyboards derived from validated visual graph plans."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping

VISUAL_STORYBOARD_SCHEMA = "codenib.visual-storyboard.v1"
VISUAL_STORYBOARD_VERSION = 1
VISUAL_STORYBOARD_MANIFEST_SCHEMA = "codenib.visual-storyboard-manifest.v1"
VISUAL_STORYBOARD_MANIFEST_VERSION = 1

_MAX_FRAMES_PER_STORYBOARD = 12
_MAX_TEXT_BYTES = 2048
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class StoryboardFrame:
    """One grounded frame that a later image/video backend can materialize."""

    id: str
    title: str
    narration: str
    visual_prompt: str
    duration_ms: int = 3500
    focus_node_ids: tuple[str, ...] = ()
    source_citations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["focus_node_ids"] = list(self.focus_node_ids)
        payload["source_citations"] = list(self.source_citations)
        return payload


@dataclass(frozen=True)
class VisualStoryboard:
    """A deterministic storyboard for explaining one repository visual asset."""

    artifact_path: str
    title: str
    frames: tuple[StoryboardFrame, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": VISUAL_STORYBOARD_SCHEMA,
            "version": VISUAL_STORYBOARD_VERSION,
            "artifact_path": self.artifact_path,
            "title": self.title,
            "frames": [frame.to_dict() for frame in self.frames],
        }
        payload["storyboard_sha256"] = _sha256_json(
            {key: value for key, value in payload.items() if key != "storyboard_sha256"}
        )
        return payload


def build_visual_storyboard_manifest(
    visual_graph_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build video-ready storyboards from validated visual graph plans."""

    storyboards = []
    for plan in visual_graph_manifest.get("plans") or ():
        if not isinstance(plan, Mapping):
            continue
        storyboard = build_visual_storyboard(plan)
        if storyboard["frames"]:
            validate_visual_storyboard(storyboard)
            storyboards.append(storyboard)
    payload = {
        "schema": VISUAL_STORYBOARD_MANIFEST_SCHEMA,
        "version": VISUAL_STORYBOARD_MANIFEST_VERSION,
        "visual_graph_manifest_sha256": str(
            visual_graph_manifest.get("manifest_sha256") or ""
        ),
        "storyboard_count": len(storyboards),
        "storyboards": sorted(
            storyboards,
            key=lambda storyboard: storyboard["artifact_path"],
        ),
    }
    payload["manifest_sha256"] = _sha256_json(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    return payload


def build_visual_storyboard(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Build one storyboard from a validated visual graph plan."""

    artifact_path = _safe_relative_path(
        plan.get("artifact_path"),
        label="artifact_path",
    )
    nodes = [node for node in plan.get("nodes") or () if isinstance(node, Mapping)]
    edges = [edge for edge in plan.get("edges") or () if isinstance(edge, Mapping)]
    node_by_id = {_safe_text(node.get("id")): node for node in nodes}
    title = _storyboard_title(artifact_path, nodes)
    frames: list[StoryboardFrame] = [
        StoryboardFrame(
            id="orient",
            title="Orient the reader",
            narration=(
                f"Start from the repository visual asset {artifact_path} and "
                "name the code elements it is trying to explain."
            ),
            visual_prompt=(
                "Show the original diagram context with the grounded code "
                "entities highlighted as readable labels."
            ),
            focus_node_ids=tuple(
                _safe_text(node.get("id")) for node in nodes[:4] if node.get("id")
            ),
            source_citations=tuple(_source_citations(nodes[:4])),
        )
    ]
    for index, edge in enumerate(edges[: max(0, _MAX_FRAMES_PER_STORYBOARD - 2)]):
        source = node_by_id.get(_safe_text(edge.get("source"))) or {}
        target = node_by_id.get(_safe_text(edge.get("target"))) or {}
        source_label = _safe_text(source.get("label")) or _safe_text(edge.get("source"))
        target_label = _safe_text(target.get("label")) or _safe_text(edge.get("target"))
        relation = _safe_text(edge.get("relation") or "related_to")
        frames.append(
            StoryboardFrame(
                id=f"relation-{index + 1}",
                title=f"{source_label} {relation} {target_label}",
                narration=(
                    f"Explain how {source_label} {relation.replace('_', ' ')} "
                    f"{target_label}, using the cited source as the evidence."
                ),
                visual_prompt=(
                    f"Animate focus from {source_label} to {target_label}; keep "
                    "surrounding nodes muted and display the relation label."
                ),
                focus_node_ids=tuple(
                    node_id
                    for node_id in (
                        _safe_text(edge.get("source")),
                        _safe_text(edge.get("target")),
                    )
                    if node_id
                ),
                source_citations=tuple(_source_citations([source, target])),
            )
        )
    frames.append(
        StoryboardFrame(
            id="grounding",
            title="Show the source grounding",
            narration=(
                "Close by showing the files and symbols that ground the visual "
                "story, so the reader can jump back to code."
            ),
            visual_prompt=(
                "Show a compact source-citation ledger beside the diagram, with "
                "file paths and symbols clearly readable."
            ),
            focus_node_ids=tuple(
                _safe_text(node.get("id")) for node in nodes[:6] if node.get("id")
            ),
            source_citations=tuple(_source_citations(nodes)),
        )
    )
    storyboard = VisualStoryboard(
        artifact_path=artifact_path,
        title=title,
        frames=tuple(frames[:_MAX_FRAMES_PER_STORYBOARD]),
    ).to_dict()
    return validate_visual_storyboard(storyboard)


def validate_visual_storyboard(
    storyboard: Mapping[str, Any],
    *,
    allowed_source_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Validate a storyboard and return a normalized dictionary."""

    if not isinstance(storyboard, Mapping):
        raise ValueError("visual storyboard must be an object")
    if storyboard.get("schema") != VISUAL_STORYBOARD_SCHEMA:
        raise ValueError("visual storyboard schema is unsupported")
    if storyboard.get("version") != VISUAL_STORYBOARD_VERSION:
        raise ValueError("visual storyboard version is unsupported")
    artifact_path = _safe_relative_path(
        storyboard.get("artifact_path"),
        label="artifact_path",
    )
    title = _safe_text(storyboard.get("title"))
    frames = [_frame_from_mapping(frame) for frame in storyboard.get("frames") or ()]
    if not frames:
        raise ValueError("visual storyboard must include frames")
    if len(frames) > _MAX_FRAMES_PER_STORYBOARD:
        raise ValueError("visual storyboard has too many frames")
    ids = [frame.id for frame in frames]
    if len(ids) != len(set(ids)):
        raise ValueError("visual storyboard frame ids must be unique")
    for frame in frames:
        for source_path in frame.source_citations:
            _safe_relative_path(source_path, label="source_citations")
            if (
                allowed_source_paths is not None
                and source_path not in allowed_source_paths
            ):
                raise ValueError("visual storyboard source citation is not allowed")
    normalized = VisualStoryboard(
        artifact_path=artifact_path,
        title=title,
        frames=tuple(frames),
    ).to_dict()
    recorded = str(storyboard.get("storyboard_sha256") or "")
    if recorded and recorded != normalized["storyboard_sha256"]:
        raise ValueError("visual storyboard hash does not match")
    return normalized


def compile_visual_storyboard_to_markdown(storyboard: Mapping[str, Any]) -> str:
    """Compile a validated storyboard to a compact Markdown shot list."""

    normalized = validate_visual_storyboard(storyboard)
    lines = [f"## {normalized['title']}", ""]
    for index, frame in enumerate(normalized["frames"], start=1):
        citations = ", ".join(frame.get("source_citations") or ())
        suffix = f" Source: {citations}." if citations else ""
        lines.append(
            f"{index}. **{frame['title']}** ({frame['duration_ms']} ms): "
            f"{frame['narration']}{suffix}"
        )
    return "\n".join(lines) + "\n"


def _frame_from_mapping(value: Any) -> StoryboardFrame:
    if not isinstance(value, Mapping):
        raise ValueError("visual storyboard frame must be an object")
    frame_id = _safe_identifier(value.get("id"), label="frame id")
    title = _safe_text(value.get("title"))
    narration = _safe_text(value.get("narration"))
    visual_prompt = _safe_text(value.get("visual_prompt"))
    if not title or not narration or not visual_prompt:
        raise ValueError("visual storyboard frame text is required")
    duration_ms = _positive_int(value.get("duration_ms")) or 3500
    if duration_ms > 60_000:
        raise ValueError("visual storyboard frame duration is too long")
    focus_node_ids = tuple(
        _safe_identifier(item, label="focus_node_ids")
        for item in value.get("focus_node_ids") or ()
        if _safe_text(item)
    )
    source_citations = tuple(
        _safe_relative_path(item, label="source_citations")
        for item in value.get("source_citations") or ()
        if _safe_text(item)
    )
    return StoryboardFrame(
        id=frame_id,
        title=title,
        narration=narration,
        visual_prompt=visual_prompt,
        duration_ms=duration_ms,
        focus_node_ids=focus_node_ids,
        source_citations=source_citations,
    )


def _storyboard_title(artifact_path: str, nodes: list[Mapping[str, Any]]) -> str:
    labels = [_safe_text(node.get("label")) for node in nodes[:3]]
    labels = [label for label in labels if label]
    if labels:
        return " → ".join(labels)
    return PurePosixPath(artifact_path).stem.replace("_", " ").replace("-", " ").title()


def _source_citations(nodes: list[Mapping[str, Any]]) -> list[str]:
    paths = []
    for node in nodes:
        path = _safe_text(node.get("source_path"))
        if path and path not in paths:
            paths.append(path)
    return paths[:8]


def _safe_relative_path(value: Any, *, label: str) -> str:
    text = _safe_text(value)
    if not text:
        return ""
    path = PurePosixPath(text.replace("\\", "/"))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"visual storyboard {label} must be repository-relative")
    return path.as_posix()


def _safe_identifier(value: Any, *, label: str) -> str:
    text = _safe_text(value)
    if not text:
        raise ValueError(f"visual storyboard {label} is required")
    if not re.match(r"^[A-Za-z0-9_.:-]+$", text):
        raise ValueError(f"visual storyboard {label} is invalid")
    return text[:96]


def _safe_text(value: Any, max_bytes: int = _MAX_TEXT_BYTES) -> str:
    text = str(value or "").strip()
    if _CONTROL_RE.search(text):
        raise ValueError("visual storyboard text contains control characters")
    encoded = text.encode("utf-8")
    if len(encoded) > max_bytes:
        text = encoded[:max_bytes].decode("utf-8", "ignore").rstrip()
    return text


def _positive_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "VISUAL_STORYBOARD_MANIFEST_SCHEMA",
    "VISUAL_STORYBOARD_MANIFEST_VERSION",
    "VISUAL_STORYBOARD_SCHEMA",
    "VISUAL_STORYBOARD_VERSION",
    "StoryboardFrame",
    "VisualStoryboard",
    "build_visual_storyboard",
    "build_visual_storyboard_manifest",
    "compile_visual_storyboard_to_markdown",
    "validate_visual_storyboard",
]
