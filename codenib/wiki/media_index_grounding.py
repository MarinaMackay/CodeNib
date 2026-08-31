# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Index-backed evidence for visual-to-code grounding."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_HINTS = 8
_MAX_RESULTS = 40
_MAX_TEXT_BYTES = 4096
_GENERIC_IDENTIFIERS = frozenset(
    {
        "assets",
        "base",
        "code",
        "data",
        "docs",
        "experiments",
        "image",
        "media",
        "preview",
        "ready",
        "replay",
        "result",
        "results",
        "test",
        "tests",
        "wiki",
    }
)


@dataclass(frozen=True, slots=True)
class _RankedEvidence:
    score: float
    evidence: str
    line: int = 0


class IndexBackedVisualGroundingScorer:
    """Score source candidates only when a CodeNib index returns evidence.

    BM25 contributes exact symbol definitions and identifier occurrences. An
    LSP-shaped provider contributes definition/reference results and therefore
    transparently benefits from CodeGraph, SCIP FactQueryIndex, or clangd's
    native query index according to the provider selected by CodeNib.
    """

    def __init__(
        self,
        *,
        bm25: Any = None,
        lsp_provider: Any = None,
        repo_path: str | Path | None = None,
        top_k: int = 20,
    ) -> None:
        if bm25 is None and lsp_provider is None:
            raise ValueError("at least one grounding index must be provided")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 40
        ):
            raise ValueError("top_k must be an integer between 1 and 40")
        self.bm25 = bm25
        self.lsp_provider = lsp_provider
        self.repo_path = (
            Path(repo_path).expanduser().resolve() if repo_path is not None else None
        )
        self.top_k = top_k
        self._cache: dict[tuple[str, ...], dict[tuple[str, str], _RankedEvidence]] = {}

    def augment_source_candidates(
        self,
        visual_facts_manifest: Mapping[str, Any],
        source_candidates: Iterable[Mapping[str, Any]],
        *,
        max_candidates: int = 8192,
    ) -> list[dict[str, Any]]:
        """Prioritize index-returned targets within the candidate budget."""

        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or not 0 <= max_candidates <= 8192
        ):
            raise ValueError("max_candidates must be an integer between 0 and 8192")
        if max_candidates == 0:
            return []
        prioritized: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        for fact_pack in _mapping_items(visual_facts_manifest.get("facts"), 4096):
            for entity in _mapping_items(fact_pack.get("entities"), 32):
                hints = _entity_hints(entity)
                if not hints:
                    continue
                evidence = self._cache.get(hints)
                if evidence is None:
                    evidence = self._collect(hints)
                    self._cache[hints] = evidence
                for (path, symbol), ranked in evidence.items():
                    candidate = {
                        "path": path,
                        "symbol": symbol,
                        "kind": "symbol" if symbol else "source",
                        "line": ranked.line if symbol else 0,
                    }
                    key = _candidate_key(candidate)
                    prioritized.setdefault(key, candidate)
                    if len(prioritized) >= max_candidates:
                        return list(prioritized.values())
        for candidate in _mapping_items(source_candidates, max_candidates):
            normalized = _source_candidate(candidate, repo_path=self.repo_path)
            if normalized is None:
                continue
            prioritized.setdefault(_candidate_key(normalized), normalized)
            if len(prioritized) >= max_candidates:
                break
        return list(prioritized.values())

    def __call__(
        self,
        entity: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        hints = _entity_hints(entity)
        if not hints:
            return None
        evidence = self._cache.get(hints)
        if evidence is None:
            evidence = self._collect(hints)
            self._cache[hints] = evidence
        path = _safe_path(candidate.get("path"), repo_path=self.repo_path)
        symbol = _safe_text(candidate.get("symbol"))
        ranked = evidence.get((path, symbol))
        if ranked is None and not symbol:
            ranked = evidence.get((path, ""))
        if ranked is None:
            return None
        visual_confidence = _confidence(entity.get("confidence"))
        score = min(ranked.score, visual_confidence)
        if score <= 0:
            return None
        return {
            "score": round(score, 4),
            "evidence": _safe_text(
                f"{ranked.evidence}; visual confidence={visual_confidence:.4f}"
            ),
        }

    def _collect(
        self, hints: tuple[str, ...]
    ) -> dict[tuple[str, str], _RankedEvidence]:
        evidence: dict[tuple[str, str], _RankedEvidence] = {}
        identifiers = _query_identifiers(hints)
        if self.lsp_provider is not None:
            for identifier in identifiers:
                self._collect_lsp(evidence, identifier)
        if self.bm25 is not None:
            for identifier in identifiers:
                self._collect_bm25(evidence, identifier, hints)
        return evidence

    def _collect_lsp(
        self,
        evidence: dict[tuple[str, str], _RankedEvidence],
        identifier: str,
    ) -> None:
        try:
            raw_definitions = self.lsp_provider.definition(
                symbol=identifier,
                top_k=min(self.top_k, 8),
            )
        except ValueError:
            raw_definitions = ()
        definitions = tuple(
            _bounded_results(
                raw_definitions,
                self.top_k,
            )
        )
        backend = _provider_backend(raw_definitions)
        definition_maximum = 1.0 if len(definitions) == 1 else 0.9
        for rank, node in enumerate(definitions):
            self._add_node(
                evidence,
                node,
                requested_symbol=identifier,
                score=_rank_score(definition_maximum, rank),
                label=f"LSP definition ({backend})",
            )
        try:
            raw_references = self.lsp_provider.references(
                symbol=identifier,
                include_declaration=True,
                top_k=self.top_k,
            )
        except ValueError:
            raw_references = ()
        backend = _provider_backend(raw_references)
        for rank, node in enumerate(_bounded_results(raw_references, self.top_k)):
            self._add_node(
                evidence,
                node,
                requested_symbol=None,
                score=_rank_score(0.85, rank),
                label=f"LSP reference ({backend})",
            )

    def _collect_bm25(
        self,
        evidence: dict[tuple[str, str], _RankedEvidence],
        identifier: str,
        hints: tuple[str, ...],
    ) -> None:
        definitions = self.bm25.search(
            query=identifier,
            top_k=self.top_k,
            return_code_content=False,
            wrap_with_ln=False,
        )
        for rank, node in enumerate(_bounded_results(definitions, self.top_k)):
            fields = _node_fields(node, repo_path=self.repo_path)
            if fields is None or _bare_symbol(fields[1]) != identifier:
                continue
            self._add_node(
                evidence,
                node,
                requested_symbol=identifier,
                score=_rank_score(0.94, rank),
                label="BM25 exact definition",
            )
        occurrences = self.bm25.search_identifier_occurrences(
            identifier,
            context_query=" ".join(hints),
            top_k=self.top_k,
            wrap_with_ln=False,
        )
        for rank, node in enumerate(_bounded_results(occurrences, self.top_k)):
            self._add_node(
                evidence,
                node,
                requested_symbol=None,
                score=_rank_score(0.72, rank),
                label="BM25 exact identifier occurrence",
            )

    def _add_node(
        self,
        evidence: dict[tuple[str, str], _RankedEvidence],
        node: Any,
        *,
        requested_symbol: str | None,
        score: float,
        label: str,
    ) -> None:
        fields = _node_fields(node, repo_path=self.repo_path)
        if fields is None:
            return
        path, node_name, line = fields
        symbol = requested_symbol or _bare_symbol(node_name)
        if not _IDENTIFIER_RE.fullmatch(symbol):
            symbol = ""
        detail = _safe_text(f"{label}; index result {path}:{symbol or node_name}")
        if symbol:
            _retain_best(evidence, (path, symbol), score, detail, line=line)
        _retain_best(evidence, (path, ""), score * 0.8, detail, line=0)


def build_index_backed_visual_grounding_scorer(
    repo_path: str | Path,
    *,
    mode: str = "bm25",
    languages: Iterable[str] = ("python",),
    cache_dir: str | Path | None = None,
) -> IndexBackedVisualGroundingScorer:
    """Load CodeNib's existing indexes for visual-to-code grounding.

    Index construction is intentionally lazy so importing the Wiki package does
    not initialize compiler or LSP dependencies. Both full and incremental
    bundle builders use this factory, keeping backend selection identical.
    """

    if mode not in {"bm25", "bm25+lsp"}:
        raise ValueError("index grounding mode must be 'bm25' or 'bm25+lsp'")
    normalized_languages = tuple(
        language.strip()
        for language in languages
        if isinstance(language, str) and language.strip()
    ) or ("python",)

    from codenib.compiler.skill_context import build_skill_contexts

    skill_ids = ["bm25_search"]
    if mode == "bm25+lsp":
        skill_ids.extend(["lsp_definition", "lsp_references"])
    contexts = build_skill_contexts(
        str(Path(repo_path).expanduser()),
        skill_ids,
        languages=normalized_languages,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        skills_dir=str(Path(__file__).resolve().parents[1] / "agent" / "skills"),
    )
    retrieve = contexts.get("retrieve")
    bm25 = getattr(retrieve, "bm25", None)
    expand = contexts.get("expand")
    lsp_provider = getattr(expand, "lsp_provider", None)
    if mode == "bm25+lsp" and lsp_provider is None:
        code_graph = getattr(expand, "code_graph", None)
        if code_graph is not None:
            from codenib.agent.lsp_provider import StaticLSPProvider

            lsp_provider = StaticLSPProvider(code_graph)
    if bm25 is None:
        raise ValueError("index-backed grounding did not load a BM25 index")
    if mode == "bm25+lsp" and lsp_provider is None:
        raise ValueError("bm25+lsp grounding did not load an LSP provider")
    return IndexBackedVisualGroundingScorer(
        bm25=bm25,
        lsp_provider=lsp_provider if mode == "bm25+lsp" else None,
        repo_path=repo_path,
    )


def _entity_hints(entity: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[Any] = [entity.get("name")]
    raw = entity.get("grounding_candidates")
    if not isinstance(raw, (str, bytes, bytearray, Mapping)):
        try:
            values.extend(islice(iter(raw or ()), _MAX_HINTS))
        except TypeError:
            pass
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        hint = _safe_text(value)
        if hint and hint not in seen:
            seen.add(hint)
            output.append(hint)
        if len(output) >= _MAX_HINTS:
            break
    return tuple(output)


def _node_fields(
    node: Any,
    *,
    repo_path: Path | None,
) -> tuple[str, str, int] | None:
    if isinstance(node, Mapping):
        value = node
    elif hasattr(node, "model_dump"):
        value = node.model_dump()
    elif hasattr(node, "dict"):
        value = node.dict()
    else:
        value = getattr(node, "__dict__", {})
    path = _safe_path(value.get("file"), repo_path=repo_path)
    if not path:
        return None
    name = _safe_text(value.get("node_name") or value.get("name"))
    start_line = value.get("start_line")
    line = (
        start_line + 1
        if isinstance(start_line, int)
        and not isinstance(start_line, bool)
        and start_line >= 0
        else 0
    )
    return path, name, line


def _query_identifiers(hints: tuple[str, ...]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        identifier = _bare_symbol(hint)
        if (
            _IDENTIFIER_RE.fullmatch(identifier)
            and identifier.lower() not in _GENERIC_IDENTIFIERS
            and identifier not in seen
        ):
            seen.add(identifier)
            output.append(identifier)
    return tuple(output)


def _safe_path(value: Any, *, repo_path: Path | None) -> str:
    text = _safe_text(value)
    if not text or "\\" in text:
        return ""
    path = Path(text)
    if path.is_absolute():
        if repo_path is None:
            return ""
        try:
            text = path.resolve().relative_to(repo_path).as_posix()
        except (OSError, ValueError):
            return ""
    pure = PurePosixPath(text)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        return ""
    return pure.as_posix()


def _bare_symbol(value: str) -> str:
    return value.split(":")[-1].split(".")[-1].split("#")[-1].rstrip("()").strip()


def _provider_backend(results: Any) -> str:
    metadata = getattr(results, "provider_metadata_dict", None)
    if callable(metadata):
        payload = metadata()
        if isinstance(payload, Mapping):
            backend = _safe_text(
                payload.get("backend")
                or payload.get("behavior_contract")
                or payload.get("provider")
                or "CodeNib index"
            )
            snapshot = _safe_text(payload.get("index_snapshot"))
            return _safe_text(f"{backend}@{snapshot}") if snapshot else backend
    return "CodeNib index"


def _rank_score(maximum: float, rank: int) -> float:
    return round(maximum * (1.0 - min(rank, _MAX_RESULTS - 1) / 100.0), 4)


def _confidence(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(confidence):
        return 0.0
    return min(1.0, max(0.0, confidence))


def _retain_best(
    evidence: dict[tuple[str, str], _RankedEvidence],
    key: tuple[str, str],
    score: float,
    detail: str,
    *,
    line: int,
) -> None:
    score = min(1.0, max(0.0, float(score)))
    if not math.isfinite(score) or score <= 0:
        return
    current = evidence.get(key)
    if current is None or score > current.score:
        evidence[key] = _RankedEvidence(
            score=round(score, 4),
            evidence=detail,
            line=line,
        )


def _source_candidate(
    value: Mapping[str, Any],
    *,
    repo_path: Path | None,
) -> dict[str, Any] | None:
    path = _safe_path(value.get("path"), repo_path=repo_path)
    if not path:
        return None
    symbol = _safe_text(value.get("symbol"))
    kind = _safe_text(value.get("kind") or ("symbol" if symbol else "source"))
    raw_line = value.get("line")
    line = (
        raw_line
        if isinstance(raw_line, int)
        and not isinstance(raw_line, bool)
        and raw_line >= 0
        else 0
    )
    return {"path": path, "symbol": symbol, "kind": kind, "line": line}


def _candidate_key(value: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(value["path"]),
        str(value["symbol"]),
        str(value["kind"]),
        int(value["line"]),
    )


def _mapping_items(value: Any, limit: int) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return ()
    try:
        iterator = iter(value or ())
    except TypeError:
        return ()
    return (item for item in islice(iterator, limit) if isinstance(item, Mapping))


def _bounded_results(value: Any, limit: int) -> Iterable[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return ()
    try:
        return tuple(islice(iter(value or ()), limit))
    except TypeError:
        return ()


def _safe_text(value: Any) -> str:
    text = str(value or "").strip()
    text = "".join(
        character
        for character in text
        if ord(character) >= 0x20 and ord(character) != 0x7F
    )
    raw = text.encode("utf-8")
    if len(raw) <= _MAX_TEXT_BYTES:
        return text
    return raw[:_MAX_TEXT_BYTES].decode("utf-8", errors="ignore").rstrip()


__all__ = ["IndexBackedVisualGroundingScorer"]
