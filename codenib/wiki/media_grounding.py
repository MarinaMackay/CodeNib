# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Ground structured visual facts to repository files and symbols."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..repository_filters import walk_repository_files
from ..repository_source_selection import (
    DEFAULT_REPOSITORY_SOURCE_SELECTION,
    RepositorySourceSelection,
)

MEDIA_GROUNDING_SCHEMA = "codenib.media-grounding.v1"
MEDIA_GROUNDING_VERSION = 1

_SOURCE_EXTENSIONS = frozenset(
    {
        ".go",
        ".java",
        ".js",
        ".jsx",
        ".kt",
        ".py",
        ".rs",
        ".scala",
        ".swift",
        ".ts",
        ".tsx",
    }
)
_MAX_SOURCE_BYTES = 1024 * 1024
_MAX_CANDIDATES = 8192
_MAX_BINDINGS_PER_ENTITY = 5
_SYMBOL_RE = re.compile(
    r"\b(?:class|def|function|const|let|var|interface|type|struct|enum)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
_CAMEL_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]{2,}\b")
VisualGroundingScorer = Callable[
    [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any] | None
]


@dataclass(frozen=True)
class SourceSymbolCandidate:
    """One repository source target that a visual entity can bind to."""

    path: str
    symbol: str = ""
    kind: str = "source"
    line: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualCodeBinding:
    """A candidate grounding from a visual entity to source evidence."""

    artifact_path: str
    entity_name: str
    source_path: str
    symbol: str = ""
    kind: str = "source"
    line: int = 0
    score: float = 0.0
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualGroundingManifest:
    """Stable visual-code binding manifest for one visual facts manifest."""

    schema: str
    version: int
    visual_facts_manifest_sha256: str
    bindings: tuple[VisualCodeBinding, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "visual_facts_manifest_sha256": self.visual_facts_manifest_sha256,
            "binding_count": len(self.bindings),
            "bindings": [binding.to_dict() for binding in self.bindings],
            "manifest_sha256": self.manifest_sha256,
        }

    @property
    def manifest_sha256(self) -> str:
        payload = {
            "schema": self.schema,
            "version": self.version,
            "visual_facts_manifest_sha256": self.visual_facts_manifest_sha256,
            "bindings": [binding.to_dict() for binding in self.bindings],
        }
        return _sha256_json(payload)


def discover_source_symbol_candidates(
    repo_path: str | Path,
    *,
    exclude_roots: Iterable[str | Path] = (),
    selection: RepositorySourceSelection = DEFAULT_REPOSITORY_SOURCE_SELECTION,
    max_candidates: int = _MAX_CANDIDATES,
) -> list[dict[str, Any]]:
    """Return a bounded, deterministic source-symbol inventory for grounding."""

    root = Path(repo_path).expanduser().resolve()
    selected = RepositorySourceSelection(selection.exclude_subtrees)
    candidates: list[SourceSymbolCandidate] = []
    seen: set[tuple[str, str, int]] = set()
    for path in walk_repository_files(
        root,
        exclude_roots=exclude_roots,
        selection=selected,
    ):
        if len(candidates) >= max(0, max_candidates):
            break
        if path.is_symlink() or path.suffix.lower() not in _SOURCE_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > _MAX_SOURCE_BYTES:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        relative = path.relative_to(root).as_posix()
        candidates.append(SourceSymbolCandidate(path=relative))
        for symbol, line in _symbols(text):
            key = (relative, symbol, line)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                SourceSymbolCandidate(
                    path=relative,
                    symbol=symbol,
                    kind="symbol",
                    line=line,
                )
            )
            if len(candidates) >= max(0, max_candidates):
                break
    return [candidate.to_dict() for candidate in candidates[: max(0, max_candidates)]]


def ground_visual_facts_to_sources(
    visual_facts_manifest: Mapping[str, Any],
    source_candidates: Iterable[Mapping[str, Any]],
    *,
    max_bindings_per_entity: int = _MAX_BINDINGS_PER_ENTITY,
    scorer: VisualGroundingScorer | None = None,
) -> dict[str, Any]:
    """Ground visual entities to a source inventory.

    The default scorer is deterministic and lexical. Callers can pass a scorer
    backed by BM25, embeddings, CodeGraph, LSP facts, or FactQueryIndex without
    changing the visual-code binding manifest schema.
    """

    candidates = [
        _candidate_from_mapping(candidate)
        for candidate in source_candidates
        if isinstance(candidate, Mapping)
    ]
    bindings: list[VisualCodeBinding] = []
    for fact_pack in visual_facts_manifest.get("facts") or ():
        if not isinstance(fact_pack, Mapping):
            continue
        artifact_path = _safe_text(fact_pack.get("artifact_path"))
        for entity in fact_pack.get("entities") or ():
            if not isinstance(entity, Mapping):
                continue
            entity_name = _safe_text(entity.get("name"))
            if not entity_name:
                continue
            hints = [
                entity_name,
                *[
                    _safe_text(candidate)
                    for candidate in entity.get("grounding_candidates") or ()
                ],
            ]
            scored = [
                binding
                for binding in (
                    _score_with_optional_scorer(
                        artifact_path=artifact_path,
                        entity=entity,
                        entity_name=entity_name,
                        hints=hints,
                        candidate=candidate,
                        scorer=scorer,
                    )
                    for candidate in candidates
                )
                if binding is not None
            ]
            scored.sort(
                key=lambda binding: (
                    -binding.score,
                    binding.source_path,
                    binding.symbol,
                    binding.line,
                )
            )
            bindings.extend(scored[: max(0, max_bindings_per_entity)])
    manifest = VisualGroundingManifest(
        schema=MEDIA_GROUNDING_SCHEMA,
        version=MEDIA_GROUNDING_VERSION,
        visual_facts_manifest_sha256=_safe_text(
            visual_facts_manifest.get("manifest_sha256")
        ),
        bindings=tuple(
            sorted(
                _dedupe_bindings(bindings),
                key=lambda binding: (
                    binding.artifact_path,
                    binding.entity_name,
                    -binding.score,
                    binding.source_path,
                    binding.symbol,
                ),
            )
        ),
    )
    return manifest.to_dict()


def _score_with_optional_scorer(
    *,
    artifact_path: str,
    entity: Mapping[str, Any],
    entity_name: str,
    hints: Iterable[str],
    candidate: SourceSymbolCandidate,
    scorer: VisualGroundingScorer | None,
) -> VisualCodeBinding | None:
    if scorer is None:
        return _score_candidate(
            artifact_path=artifact_path,
            entity_name=entity_name,
            hints=hints,
            candidate=candidate,
        )
    raw = scorer(entity, candidate.to_dict())
    if not isinstance(raw, Mapping):
        return None
    score = _confidence(raw.get("score"))
    if score <= 0:
        return None
    return VisualCodeBinding(
        artifact_path=artifact_path,
        entity_name=entity_name,
        source_path=candidate.path,
        symbol=candidate.symbol,
        kind=candidate.kind,
        line=candidate.line,
        score=round(score, 4),
        evidence=_safe_text(raw.get("evidence") or "custom scorer"),
    )


def _score_candidate(
    *,
    artifact_path: str,
    entity_name: str,
    hints: Iterable[str],
    candidate: SourceSymbolCandidate,
) -> VisualCodeBinding | None:
    normalized_hints = [_normalize(hint) for hint in hints if _normalize(hint)]
    candidate_symbol = _normalize(candidate.symbol)
    candidate_path = _normalize(Path(candidate.path).stem)
    candidate_full_path = _normalize(candidate.path)
    score = 0.0
    evidence = ""
    for hint in normalized_hints:
        if candidate_symbol and hint == candidate_symbol:
            score = max(score, 1.0)
            evidence = "exact symbol match"
        elif candidate_symbol and (
            hint in candidate_symbol or candidate_symbol in hint
        ):
            score = max(score, 0.75)
            evidence = "partial symbol match"
        elif hint == candidate_path:
            score = max(score, 0.6)
            evidence = "file stem match"
        elif hint in candidate_full_path:
            score = max(score, 0.45)
            evidence = "path match"
    if score <= 0:
        return None
    return VisualCodeBinding(
        artifact_path=artifact_path,
        entity_name=entity_name,
        source_path=candidate.path,
        symbol=candidate.symbol,
        kind=candidate.kind,
        line=candidate.line,
        score=round(score, 4),
        evidence=evidence,
    )


def _symbols(text: str) -> Iterable[tuple[str, int]]:
    seen = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        for regex in (_SYMBOL_RE, _CAMEL_RE):
            for match in regex.finditer(line):
                symbol = match.group(1) if regex is _SYMBOL_RE else match.group(0)
                if symbol in seen:
                    continue
                seen.add(symbol)
                yield symbol, line_number


def _candidate_from_mapping(value: Mapping[str, Any]) -> SourceSymbolCandidate:
    return SourceSymbolCandidate(
        path=_safe_text(value.get("path")),
        symbol=_safe_text(value.get("symbol")),
        kind=_safe_text(value.get("kind") or "source"),
        line=_positive_int(value.get("line")),
    )


def _dedupe_bindings(
    bindings: Iterable[VisualCodeBinding],
) -> tuple[VisualCodeBinding, ...]:
    best: dict[tuple[str, str, str, str, int], VisualCodeBinding] = {}
    for binding in bindings:
        key = (
            binding.artifact_path,
            binding.entity_name,
            binding.source_path,
            binding.symbol,
            binding.line,
        )
        previous = best.get(key)
        if previous is None or binding.score > previous.score:
            best[key] = binding
    return tuple(best.values())


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _positive_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _confidence(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, confidence))


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


__all__ = [
    "MEDIA_GROUNDING_SCHEMA",
    "MEDIA_GROUNDING_VERSION",
    "SourceSymbolCandidate",
    "VisualGroundingScorer",
    "VisualCodeBinding",
    "VisualGroundingManifest",
    "discover_source_symbol_candidates",
    "ground_visual_facts_to_sources",
]
