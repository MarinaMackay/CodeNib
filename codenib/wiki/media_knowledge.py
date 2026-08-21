# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Queryable multimodal repository knowledge view for wiki media."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

MULTIMODAL_KNOWLEDGE_SCHEMA = "codenib.multimodal-knowledge-view.v1"
MULTIMODAL_KNOWLEDGE_VERSION = 1


@dataclass(frozen=True)
class MultimodalKnowledgeView:
    """A compact, persistent view joining media artifacts, facts, and grounding."""

    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def search_visual_context(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        return search_visual_context(self.payload, query, limit=limit)

    def get_visual_evidence(self, artifact_path: str) -> dict[str, Any] | None:
        return get_visual_evidence(self.payload, artifact_path)

    def find_visual_code_links(
        self,
        source_path: str,
        *,
        symbol: str = "",
    ) -> list[dict[str, Any]]:
        return find_visual_code_links(self.payload, source_path, symbol=symbol)


def build_multimodal_knowledge_view(
    media_manifest: Mapping[str, Any],
    visual_facts_manifest: Mapping[str, Any],
    grounding_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Join media artifacts, visual facts, and source bindings into one view."""

    artifacts = {
        str(artifact.get("path") or ""): dict(artifact)
        for artifact in media_manifest.get("artifacts") or ()
        if isinstance(artifact, Mapping) and artifact.get("path")
    }
    facts = {
        str(fact.get("artifact_path") or ""): dict(fact)
        for fact in visual_facts_manifest.get("facts") or ()
        if isinstance(fact, Mapping) and fact.get("artifact_path")
    }
    bindings_by_artifact: dict[str, list[dict[str, Any]]] = {}
    for binding in grounding_manifest.get("bindings") or ():
        if not isinstance(binding, Mapping):
            continue
        artifact_path = str(binding.get("artifact_path") or "")
        if artifact_path:
            bindings_by_artifact.setdefault(artifact_path, []).append(dict(binding))

    entries = []
    for path in sorted(set(artifacts) | set(facts) | set(bindings_by_artifact)):
        artifact = artifacts.get(path, {})
        fact = facts.get(path, {})
        bindings = sorted(
            bindings_by_artifact.get(path, []),
            key=lambda item: (
                str(item.get("source_path") or ""),
                str(item.get("symbol") or ""),
                str(item.get("entity_name") or ""),
            ),
        )
        entries.append(
            {
                "artifact": artifact,
                "facts": fact,
                "bindings": bindings,
                "search_text": _entry_search_text(artifact, fact, bindings),
            }
        )
    payload = {
        "schema": MULTIMODAL_KNOWLEDGE_SCHEMA,
        "version": MULTIMODAL_KNOWLEDGE_VERSION,
        "media_manifest_sha256": str(media_manifest.get("manifest_sha256") or ""),
        "visual_facts_manifest_sha256": str(
            visual_facts_manifest.get("manifest_sha256") or ""
        ),
        "grounding_manifest_sha256": str(
            grounding_manifest.get("manifest_sha256") or ""
        ),
        "entry_count": len(entries),
        "entries": entries,
    }
    payload["view_sha256"] = _sha256_json(
        {key: value for key, value in payload.items() if key != "view_sha256"}
    )
    return payload


def search_visual_context(
    view: Mapping[str, Any],
    query: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search multimodal entries using a deterministic lexical scorer."""

    tokens = _tokens(query)
    if not tokens:
        return []
    results = []
    for entry in view.get("entries") or ():
        if not isinstance(entry, Mapping):
            continue
        haystack = str(entry.get("search_text") or "").lower()
        score = sum(1 for token in tokens if token in haystack)
        if score:
            results.append(
                {
                    "artifact_path": ((entry.get("artifact") or {}).get("path") or ""),
                    "score": score,
                    "artifact": dict(entry.get("artifact") or {}),
                    "facts": dict(entry.get("facts") or {}),
                    "bindings": list(entry.get("bindings") or ()),
                }
            )
    results.sort(key=lambda item: (-item["score"], item["artifact_path"]))
    return results[: max(0, limit)]


def get_visual_evidence(
    view: Mapping[str, Any],
    artifact_path: str,
) -> dict[str, Any] | None:
    """Return one visual knowledge entry by artifact path."""

    for entry in view.get("entries") or ():
        if not isinstance(entry, Mapping):
            continue
        artifact = entry.get("artifact") or {}
        if isinstance(artifact, Mapping) and artifact.get("path") == artifact_path:
            return {
                "artifact": dict(artifact),
                "facts": dict(entry.get("facts") or {}),
                "bindings": list(entry.get("bindings") or ()),
            }
    return None


def find_visual_code_links(
    view: Mapping[str, Any],
    source_path: str,
    *,
    symbol: str = "",
) -> list[dict[str, Any]]:
    """Return visual entries that ground to a file, optionally a symbol."""

    links = []
    for entry in view.get("entries") or ():
        if not isinstance(entry, Mapping):
            continue
        for binding in entry.get("bindings") or ():
            if not isinstance(binding, Mapping):
                continue
            if binding.get("source_path") != source_path:
                continue
            if symbol and binding.get("symbol") != symbol:
                continue
            links.append(
                {
                    "artifact_path": ((entry.get("artifact") or {}).get("path") or ""),
                    "binding": dict(binding),
                    "artifact": dict(entry.get("artifact") or {}),
                    "facts": dict(entry.get("facts") or {}),
                }
            )
    links.sort(
        key=lambda item: (
            str(item["artifact_path"]),
            str(item["binding"].get("entity_name") or ""),
            str(item["binding"].get("symbol") or ""),
        )
    )
    return links


def _entry_search_text(
    artifact: Mapping[str, Any],
    fact: Mapping[str, Any],
    bindings: list[Mapping[str, Any]],
) -> str:
    parts = [
        artifact.get("path"),
        artifact.get("role_hint"),
        artifact.get("caption"),
        artifact.get("surrounding_text"),
    ]
    for entity in fact.get("entities") or ():
        if isinstance(entity, Mapping):
            parts.extend(
                [entity.get("name"), entity.get("type"), entity.get("evidence")]
            )
    for claim in fact.get("claims") or ():
        if isinstance(claim, Mapping):
            parts.extend([claim.get("text"), claim.get("evidence")])
    for binding in bindings:
        parts.extend(
            [
                binding.get("entity_name"),
                binding.get("source_path"),
                binding.get("symbol"),
                binding.get("evidence"),
            ]
        )
    return " ".join(str(part or "") for part in parts)


def _tokens(value: str) -> list[str]:
    return [
        token
        for token in re.split(r"[^a-z0-9_]+", str(value or "").lower())
        if len(token) >= 2
    ]


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
    "MULTIMODAL_KNOWLEDGE_SCHEMA",
    "MULTIMODAL_KNOWLEDGE_VERSION",
    "MultimodalKnowledgeView",
    "build_multimodal_knowledge_view",
    "find_visual_code_links",
    "get_visual_evidence",
    "search_visual_context",
]
