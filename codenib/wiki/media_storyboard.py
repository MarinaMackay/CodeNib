# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Validated, video-ready storyboards derived from visual graph plans."""

from __future__ import annotations

import hashlib
import hmac
import html
import io
import json
import os
import re
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .._bounded_json import validate_bounded_json_stream, validate_json_complexity
from ._safe_file_reads import read_regular_bytes
from .media_graph_plan import validate_visual_graph_manifest, validate_visual_graph_plan

VISUAL_STORYBOARD_SCHEMA = "codenib.visual-storyboard.v1"
VISUAL_STORYBOARD_VERSION = 1
VISUAL_STORYBOARD_MANIFEST_SCHEMA = "codenib.visual-storyboard-manifest.v1"
VISUAL_STORYBOARD_MANIFEST_VERSION = 1

_MAX_STORYBOARDS = 4096
_MAX_FRAMES = 12
_MAX_FOCUS_NODES = 12
_MAX_CITATIONS = 16
_MAX_DURATION_MS = 60_000
_MAX_TOTAL_DURATION_MS = 5 * 60_000
_MAX_LINE = 100_000_000
_MAX_TEXT_BYTES = 4096
_MIN_CITATION_SCORE = 0.8
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_MANIFEST_NODES = 1_000_000
_MAX_MANIFEST_TOKENS = 2_000_000
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]{0,95}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_STORYBOARD_FIELDS = frozenset(
    {
        "schema",
        "version",
        "artifact_path",
        "graph_plan_sha256",
        "title",
        "total_duration_ms",
        "frames",
        "storyboard_sha256",
    }
)
_FRAME_FIELDS = frozenset(
    {
        "id",
        "kind",
        "title",
        "narration",
        "visual_prompt",
        "duration_ms",
        "focus_node_ids",
        "source_citations",
    }
)
_CITATION_FIELDS = frozenset({"source_path", "symbol", "line"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "version",
        "visual_graph_manifest_sha256",
        "storyboard_count",
        "storyboards",
        "manifest_sha256",
    }
)


def build_visual_storyboard(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Build a deterministic shot list from one validated graph plan."""

    graph = validate_visual_graph_plan(plan)
    nodes = graph["nodes"]
    node_by_id = {node["id"]: node for node in nodes}
    frames = [_overview_frame(graph)]
    relation_limit = max(0, _MAX_FRAMES - 2)
    for index, edge in enumerate(graph["edges"][:relation_limit], start=1):
        frames.append(_relation_frame(index, edge, node_by_id))
    if not graph["edges"]:
        for index, node in enumerate(nodes[:relation_limit], start=1):
            frames.append(_node_frame(index, node))
    frames.append(_source_frame(nodes))
    frames = frames[:_MAX_FRAMES]
    payload: dict[str, Any] = {
        "schema": VISUAL_STORYBOARD_SCHEMA,
        "version": VISUAL_STORYBOARD_VERSION,
        "artifact_path": graph["artifact_path"],
        "graph_plan_sha256": graph["plan_sha256"],
        "title": _storyboard_title(graph),
        "total_duration_ms": sum(frame["duration_ms"] for frame in frames),
        "frames": frames,
    }
    payload["storyboard_sha256"] = _sha256_json(payload)
    return validate_visual_storyboard(
        payload,
        allowed_node_ids=set(node_by_id),
        allowed_citations=_citation_keys(nodes),
    )


def build_visual_storyboard_manifest(
    visual_graph_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a hash-bound storyboard sidecar from a graph manifest."""

    graph_manifest = validate_visual_graph_manifest(visual_graph_manifest)
    storyboards = [build_visual_storyboard(plan) for plan in graph_manifest["plans"]]
    payload: dict[str, Any] = {
        "schema": VISUAL_STORYBOARD_MANIFEST_SCHEMA,
        "version": VISUAL_STORYBOARD_MANIFEST_VERSION,
        "visual_graph_manifest_sha256": graph_manifest["manifest_sha256"],
        "storyboard_count": len(storyboards),
        "storyboards": storyboards,
    }
    payload["manifest_sha256"] = _sha256_json(payload)
    return validate_visual_storyboard_manifest(
        payload, visual_graph_manifest=graph_manifest
    )


def validate_visual_storyboard(
    storyboard: Mapping[str, Any],
    *,
    allowed_node_ids: set[str] | None = None,
    allowed_citations: set[tuple[str, str, int]] | None = None,
) -> dict[str, Any]:
    """Validate and normalize one persisted storyboard."""

    data = _mapping(storyboard, label="visual storyboard")
    _exact_fields(data, _STORYBOARD_FIELDS, label="visual storyboard")
    if data["schema"] != VISUAL_STORYBOARD_SCHEMA:
        raise ValueError("visual storyboard schema is unsupported")
    if type(data["version"]) is not int or data["version"] != VISUAL_STORYBOARD_VERSION:
        raise ValueError("visual storyboard version is unsupported")
    frames = [
        _validated_frame(frame)
        for frame in _mapping_items(
            data["frames"], label="visual storyboard frames", limit=_MAX_FRAMES
        )
    ]
    if not frames:
        raise ValueError("visual storyboard must contain at least one frame")
    frame_ids = [frame["id"] for frame in frames]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("visual storyboard frame ids must be unique")
    total_duration = sum(frame["duration_ms"] for frame in frames)
    if (
        type(data["total_duration_ms"]) is not int
        or data["total_duration_ms"] != total_duration
        or total_duration > _MAX_TOTAL_DURATION_MS
    ):
        raise ValueError("visual storyboard total_duration_ms is invalid")
    normalized_nodes = (
        {_identifier(value, label="allowed node id") for value in allowed_node_ids}
        if allowed_node_ids is not None
        else None
    )
    normalized_citations = (
        {
            (
                _relative_path(path, label="allowed citation path"),
                _single_line(symbol, label="allowed citation symbol"),
                _line(line),
            )
            for path, symbol, line in allowed_citations
        }
        if allowed_citations is not None
        else None
    )
    for frame in frames:
        if normalized_nodes is not None and not set(frame["focus_node_ids"]).issubset(
            normalized_nodes
        ):
            raise ValueError("visual storyboard references an unknown graph node")
        if normalized_citations is not None:
            for citation in frame["source_citations"]:
                if _citation_key(citation) not in normalized_citations:
                    raise ValueError("visual storyboard source citation is not allowed")
    normalized: dict[str, Any] = {
        "schema": VISUAL_STORYBOARD_SCHEMA,
        "version": VISUAL_STORYBOARD_VERSION,
        "artifact_path": _relative_path(data["artifact_path"], label="artifact_path"),
        "graph_plan_sha256": _digest(
            data["graph_plan_sha256"], label="graph_plan_sha256"
        ),
        "title": _required_text(data["title"], label="storyboard title"),
        "total_duration_ms": total_duration,
        "frames": frames,
    }
    expected = _sha256_json(normalized)
    recorded = _digest(data["storyboard_sha256"], label="storyboard_sha256")
    if not hmac.compare_digest(recorded, expected):
        raise ValueError("visual storyboard hash does not match")
    normalized["storyboard_sha256"] = recorded
    return normalized


def validate_visual_storyboard_manifest(
    manifest: Mapping[str, Any],
    *,
    visual_graph_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and normalize a persisted storyboard manifest."""

    data = _mapping(manifest, label="visual storyboard manifest")
    _exact_fields(data, _MANIFEST_FIELDS, label="visual storyboard manifest")
    if data["schema"] != VISUAL_STORYBOARD_MANIFEST_SCHEMA:
        raise ValueError("visual storyboard manifest schema is unsupported")
    if (
        type(data["version"]) is not int
        or data["version"] != VISUAL_STORYBOARD_MANIFEST_VERSION
    ):
        raise ValueError("visual storyboard manifest version is unsupported")
    graph_manifest = (
        validate_visual_graph_manifest(visual_graph_manifest)
        if visual_graph_manifest is not None
        else None
    )
    graph_by_path = (
        {plan["artifact_path"]: plan for plan in graph_manifest["plans"]}
        if graph_manifest is not None
        else {}
    )
    storyboards = []
    for value in _mapping_items(
        data["storyboards"],
        label="visual storyboard manifest storyboards",
        limit=_MAX_STORYBOARDS,
    ):
        preliminary = validate_visual_storyboard(value)
        plan = graph_by_path.get(preliminary["artifact_path"])
        if graph_manifest is not None and plan is None:
            raise ValueError("visual storyboard has no matching graph plan")
        if plan is not None:
            if preliminary["graph_plan_sha256"] != plan["plan_sha256"]:
                raise ValueError("visual storyboard graph plan hash does not match")
            preliminary = validate_visual_storyboard(
                value,
                allowed_node_ids={node["id"] for node in plan["nodes"]},
                allowed_citations=_citation_keys(plan["nodes"]),
            )
        storyboards.append(preliminary)
    if type(data["storyboard_count"]) is not int or data["storyboard_count"] != len(
        storyboards
    ):
        raise ValueError("visual storyboard manifest storyboard_count is invalid")
    paths = [storyboard["artifact_path"] for storyboard in storyboards]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("visual storyboard manifest paths must be unique and sorted")
    normalized: dict[str, Any] = {
        "schema": VISUAL_STORYBOARD_MANIFEST_SCHEMA,
        "version": VISUAL_STORYBOARD_MANIFEST_VERSION,
        "visual_graph_manifest_sha256": _digest(
            data["visual_graph_manifest_sha256"],
            label="visual_graph_manifest_sha256",
        ),
        "storyboard_count": len(storyboards),
        "storyboards": storyboards,
    }
    if graph_manifest is not None:
        if (
            normalized["visual_graph_manifest_sha256"]
            != graph_manifest["manifest_sha256"]
        ):
            raise ValueError("visual storyboard graph manifest hash does not match")
        if paths != list(graph_by_path):
            raise ValueError(
                "visual storyboard manifest does not cover every graph plan"
            )
    expected = _sha256_json(normalized)
    recorded = _digest(data["manifest_sha256"], label="manifest_sha256")
    if not hmac.compare_digest(recorded, expected):
        raise ValueError("visual storyboard manifest hash does not match")
    normalized["manifest_sha256"] = recorded
    validate_json_complexity(
        normalized, label="visual storyboard manifest", max_nodes=_MAX_MANIFEST_NODES
    )
    if len(_canonical_json_bytes(normalized)) > _MAX_MANIFEST_BYTES:
        raise ValueError("visual storyboard manifest exceeds the byte limit")
    return normalized


def compile_visual_storyboard_to_markdown(storyboard: Mapping[str, Any]) -> str:
    """Compile a validated storyboard into an inspectable production shot list."""

    normalized = validate_visual_storyboard(storyboard)
    lines = [f"## {_markdown_text(normalized['title'])}", ""]
    for index, frame in enumerate(normalized["frames"], start=1):
        lines.extend(
            [
                f"### {index}. {_markdown_text(frame['title'])}",
                "",
                f"- Kind: {_markdown_code(frame['kind'])}",
                f"- Duration: `{frame['duration_ms']} ms`",
                f"- Narration: {_markdown_text(frame['narration'])}",
                f"- Visual direction: {_markdown_text(frame['visual_prompt'])}",
            ]
        )
        if frame["source_citations"]:
            lines.append("- Sources:")
            for citation in frame["source_citations"]:
                suffix = f":{citation['line']}" if citation["line"] else ""
                source = _markdown_code(f"{citation['source_path']}{suffix}")
                symbol = (
                    f" ({_markdown_code(citation['symbol'])})"
                    if citation["symbol"]
                    else ""
                )
                lines.append(f"  - {source}{symbol}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_visual_storyboard_manifest(
    manifest: Mapping[str, Any], path: str | Path
) -> None:
    """Atomically persist a validated storyboard manifest."""

    normalized = validate_visual_storyboard_manifest(manifest)
    payload = (
        json.dumps(
            normalized, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise ValueError("visual storyboard manifest exceeds the byte limit")
    _atomic_write(Path(path).expanduser(), payload)


def load_visual_storyboard_manifest(path: str | Path) -> dict[str, Any]:
    """Load a bounded regular JSON storyboard manifest."""

    raw = read_regular_bytes(Path(path).expanduser(), max_bytes=_MAX_MANIFEST_BYTES)
    if raw is None:
        raise ValueError("storyboard manifest must be a stable bounded regular file")
    validate_bounded_json_stream(
        io.BytesIO(raw),
        label="visual storyboard manifest",
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
        raise ValueError("visual storyboard manifest contains invalid JSON") from exc
    return validate_visual_storyboard_manifest(decoded)


def _overview_frame(graph: Mapping[str, Any]) -> dict[str, Any]:
    nodes = graph["nodes"][:4]
    labels = ", ".join(node["label"] for node in nodes)
    return {
        "id": "overview",
        "kind": "overview",
        "title": "Orient the reader",
        "narration": f"Introduce the code elements represented by {graph['artifact_path']}.",
        "visual_prompt": (
            f"Show the original repository visual and highlight {labels}. "
            "Keep every source label readable and avoid decorative elements."
        ),
        "duration_ms": 4000,
        "focus_node_ids": [node["id"] for node in nodes],
        "source_citations": _citations(nodes),
    }


def _relation_frame(
    index: int, edge: Mapping[str, Any], node_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    source = node_by_id[edge["source"]]
    target = node_by_id[edge["target"]]
    relation = edge["relation"].replace("_", " ")
    citations = _citations([source, target])
    evidence_direction = (
        "using the cited source evidence"
        if citations
        else "while marking source grounding as pending"
    )
    return {
        "id": f"relation-{index}",
        "kind": "relation",
        "title": f"{source['label']} {relation} {target['label']}",
        "narration": (
            f"Explain how {source['label']} {relation} {target['label']} using "
            f"only the graph relation {evidence_direction}."
        ),
        "visual_prompt": (
            f"Animate focus from {source['label']} to {target['label']}; mute "
            f"other nodes and display the relation label {edge['relation']}."
        ),
        "duration_ms": 4500,
        "focus_node_ids": [source["id"], target["id"]],
        "source_citations": citations,
    }


def _node_frame(index: int, node: Mapping[str, Any]) -> dict[str, Any]:
    grounding = (
        f"Ground the explanation in {node['source_path']} and {node['symbol']}."
        if _is_citable_node(node)
        else "Mark this entity as not yet sufficiently source-grounded."
    )
    return {
        "id": f"entity-{index}",
        "kind": "entity",
        "title": node["label"],
        "narration": f"Introduce {node['label']} without inferring unseen relations.",
        "visual_prompt": (
            f"Focus on {node['label']} as a single highlighted entity. {grounding}"
        ),
        "duration_ms": 3000,
        "focus_node_ids": [node["id"]],
        "source_citations": _citations([node]),
    }


def _source_frame(nodes: list[Mapping[str, Any]]) -> dict[str, Any]:
    grounded = [node for node in nodes if _is_citable_node(node)]
    if grounded:
        narration = (
            "Close with the files, symbols, and lines that ground this explanation."
        )
        visual_prompt = (
            "Show a compact source ledger beside the visual. Keep paths, symbols, "
            "and line numbers readable; do not add uncited files."
        )
    else:
        narration = "Close by stating that high-confidence source grounding is pending."
        visual_prompt = (
            "Show a visible grounding-pending marker; do not display or invent source "
            "paths, symbols, or line numbers."
        )
    return {
        "id": "sources",
        "kind": "source_ledger",
        "title": "Verify in source",
        "narration": narration,
        "visual_prompt": visual_prompt,
        "duration_ms": 3500,
        "focus_node_ids": [node["id"] for node in grounded[:_MAX_FOCUS_NODES]],
        "source_citations": _citations(grounded),
    }


def _storyboard_title(graph: Mapping[str, Any]) -> str:
    labels = [node["label"] for node in graph["nodes"][:3]]
    if labels:
        return " → ".join(labels)
    return PurePosixPath(graph["artifact_path"]).stem.replace("_", " ").title()


def _citations(nodes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    citations = []
    seen = set()
    for node in nodes:
        if not _is_citable_node(node):
            continue
        citation = {
            "source_path": node["source_path"],
            "symbol": node["symbol"],
            "line": node["line"],
        }
        key = _citation_key(citation)
        if key not in seen:
            seen.add(key)
            citations.append(citation)
        if len(citations) >= _MAX_CITATIONS:
            break
    return citations


def _citation_keys(nodes: Iterable[Mapping[str, Any]]) -> set[tuple[str, str, int]]:
    return {
        (node["source_path"], node["symbol"], node["line"])
        for node in nodes
        if _is_citable_node(node)
    }


def _is_citable_node(node: Mapping[str, Any]) -> bool:
    return bool(node["source_path"]) and node["grounding_score"] >= _MIN_CITATION_SCORE


def _citation_key(citation: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        citation["source_path"],
        citation["symbol"],
        citation["line"],
    )


def _validated_frame(value: Any) -> dict[str, Any]:
    frame = _mapping(value, label="visual storyboard frame")
    _exact_fields(frame, _FRAME_FIELDS, label="visual storyboard frame")
    focus = [
        _identifier(item, label="focus node id")
        for item in _text_items(
            frame["focus_node_ids"],
            label="visual storyboard focus nodes",
            limit=_MAX_FOCUS_NODES,
        )
    ]
    if len(focus) != len(set(focus)):
        raise ValueError("visual storyboard focus node ids must be unique")
    citations = [
        _validated_citation(item)
        for item in _mapping_items(
            frame["source_citations"],
            label="visual storyboard source citations",
            limit=_MAX_CITATIONS,
        )
    ]
    citation_keys = [_citation_key(citation) for citation in citations]
    if len(citation_keys) != len(set(citation_keys)):
        raise ValueError("visual storyboard source citations must be unique")
    duration = frame["duration_ms"]
    if type(duration) is not int or not 1 <= duration <= _MAX_DURATION_MS:
        raise ValueError("visual storyboard frame duration_ms is invalid")
    return {
        "id": _identifier(frame["id"], label="frame id"),
        "kind": _identifier(frame["kind"], label="frame kind"),
        "title": _required_text(frame["title"], label="frame title"),
        "narration": _required_text(frame["narration"], label="frame narration"),
        "visual_prompt": _required_text(
            frame["visual_prompt"], label="frame visual_prompt"
        ),
        "duration_ms": duration,
        "focus_node_ids": focus,
        "source_citations": citations,
    }


def _validated_citation(value: Any) -> dict[str, Any]:
    citation = _mapping(value, label="visual storyboard source citation")
    _exact_fields(citation, _CITATION_FIELDS, label="visual storyboard citation")
    return {
        "source_path": _relative_path(citation["source_path"], label="source path"),
        "symbol": _single_line(citation["symbol"], label="source symbol"),
        "line": _line(citation["line"]),
    }


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
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


def _text_items(value: Any, *, label: str, limit: int) -> Iterable[str]:
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"{label} must be a bounded array")
    if any(type(item) is not str for item in value):
        raise ValueError(f"{label} must contain only text")
    return value


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields are invalid")


def _relative_path(value: Any, *, label: str) -> str:
    text = _required_text(value, label=label)
    if "\n" in text or "\r" in text:
        raise ValueError(f"visual storyboard {label} must be one line")
    path = PurePosixPath(text)
    if (
        text == "."
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != text
        or "\\" in text
    ):
        raise ValueError(f"visual storyboard {label} must be repository-relative")
    return text


def _identifier(value: Any, *, label: str) -> str:
    text = _single_line(value, label=label)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        raise ValueError(f"visual storyboard {label} is invalid")
    return text


def _single_line(value: Any, *, label: str) -> str:
    text = _text(value, label=label)
    if "\n" in text or "\r" in text:
        raise ValueError(f"visual storyboard {label} must be one line")
    return text


def _required_text(value: Any, *, label: str) -> str:
    text = _text(value, label=label)
    if not text:
        raise ValueError(f"visual storyboard {label} is required")
    return text


def _text(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"visual storyboard {label} must be text")
    text = value.strip()
    if _CONTROL_RE.search(text):
        raise ValueError(f"visual storyboard {label} contains control characters")
    if len(text.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise ValueError(f"visual storyboard {label} exceeds the byte limit")
    return text


def _line(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_LINE:
        raise ValueError("visual storyboard source line is invalid")
    return value


def _markdown_text(value: str) -> str:
    escaped = html.escape(value, quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+.!|>-])", r"\\\1", escaped)


def _markdown_code(value: str) -> str:
    return f"`{html.escape(value, quote=False).replace('`', '&#96;')}`"


def _digest(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"visual storyboard {label} is invalid")
    return value


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _atomic_write(destination: Path, payload: bytes) -> None:
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


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_number(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError("non-finite JSON number is not allowed")
    return number


__all__ = [
    "VISUAL_STORYBOARD_MANIFEST_SCHEMA",
    "VISUAL_STORYBOARD_MANIFEST_VERSION",
    "VISUAL_STORYBOARD_SCHEMA",
    "VISUAL_STORYBOARD_VERSION",
    "build_visual_storyboard",
    "build_visual_storyboard_manifest",
    "compile_visual_storyboard_to_markdown",
    "load_visual_storyboard_manifest",
    "save_visual_storyboard_manifest",
    "validate_visual_storyboard",
    "validate_visual_storyboard_manifest",
]
