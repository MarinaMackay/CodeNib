# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from codenib.wiki.media_index_grounding import IndexBackedVisualGroundingScorer


class _Results(list):
    def provider_metadata_dict(self):
        return {
            "backend": "native-scip-fact-query-index-v1",
            "index_snapshot": "fact-query:sha256:test",
        }


class _LSP:
    def __init__(self):
        self.calls = []

    def definition(self, **arguments):
        self.calls.append(("definition", arguments))
        return _Results(
            [
                SimpleNamespace(
                    file="src/compiler.py",
                    node_name="src/compiler.py:IndexCompiler",
                    start_line=20,
                )
            ]
        )

    def references(self, **arguments):
        self.calls.append(("references", arguments))
        return _Results(
            [
                SimpleNamespace(
                    file="src/runtime.py",
                    node_name="src/runtime.py:compile_repository",
                )
            ]
        )


class _BM25:
    def __init__(self):
        self.calls = []

    def search(self, **arguments):
        self.calls.append(("search", arguments))
        return [
            SimpleNamespace(
                file="src/compiler.py",
                node_name="src/compiler.py:IndexCompiler",
            ),
            SimpleNamespace(file="src/unrelated.py", node_name="Unrelated"),
        ]

    def search_identifier_occurrences(self, identifier, **arguments):
        self.calls.append(("occurrences", {"identifier": identifier, **arguments}))
        return [
            SimpleNamespace(
                file="src/runtime.py",
                node_name="src/runtime.py:compile_repository",
            )
        ]


def _entity():
    return {
        "name": "IndexCompiler",
        "type": "component",
        "evidence": "architecture label",
        "confidence": 0.9,
        "grounding_candidates": ["IndexCompiler"],
    }


def test_index_grounding_prefers_fact_query_definition_and_caches_queries():
    lsp = _LSP()
    bm25 = _BM25()
    scorer = IndexBackedVisualGroundingScorer(bm25=bm25, lsp_provider=lsp)

    definition = scorer(
        _entity(),
        {"path": "src/compiler.py", "symbol": "IndexCompiler"},
    )
    reference = scorer(
        _entity(),
        {"path": "src/runtime.py", "symbol": "compile_repository"},
    )
    missing = scorer(
        _entity(),
        {"path": "src/unrelated.py", "symbol": "Unrelated"},
    )

    assert definition == {
        "score": 0.9,
        "evidence": (
            "LSP definition (native-scip-fact-query-index-v1@"
            "fact-query:sha256:test); index result "
            "src/compiler.py:IndexCompiler; visual confidence=0.9000"
        ),
    }
    assert reference["score"] == 0.85
    assert "LSP reference" in reference["evidence"]
    assert missing is None
    assert len(lsp.calls) == 2
    assert len(bm25.calls) == 2


def test_bm25_grounding_requires_exact_returned_definition_or_occurrence():
    scorer = IndexBackedVisualGroundingScorer(bm25=_BM25())

    definition = scorer(
        _entity(),
        {"path": "src/compiler.py", "symbol": "IndexCompiler"},
    )
    occurrence = scorer(
        _entity(),
        {"path": "src/runtime.py", "symbol": "compile_repository"},
    )
    occurrence_file = scorer(
        _entity(),
        {"path": "src/runtime.py", "symbol": ""},
    )
    unsupported_symbol = scorer(
        _entity(),
        {"path": "src/runtime.py", "symbol": "UnrelatedSymbol"},
    )
    unrelated = scorer(
        _entity(),
        {"path": "src/unrelated.py", "symbol": "Unrelated"},
    )

    assert definition["score"] == 0.9
    assert definition["evidence"].startswith("BM25 exact definition")
    assert occurrence["score"] == 0.72
    assert occurrence_file["score"] == 0.576
    assert unsupported_symbol is None
    assert unrelated is None


def test_index_grounding_normalizes_repo_bound_absolute_paths(tmp_path):
    class AbsoluteBM25(_BM25):
        def search(self, **arguments):
            return [
                SimpleNamespace(
                    file=str(tmp_path / "src" / "compiler.py"),
                    node_name="IndexCompiler",
                )
            ]

        def search_identifier_occurrences(self, identifier, **arguments):
            return []

    scorer = IndexBackedVisualGroundingScorer(bm25=AbsoluteBM25(), repo_path=tmp_path)

    assert (
        scorer(
            _entity(),
            {"path": "src/compiler.py", "symbol": "IndexCompiler"},
        )["score"]
        == 0.9
    )


def test_index_grounding_rejects_missing_backends_and_invalid_limits():
    with pytest.raises(ValueError, match="at least one"):
        IndexBackedVisualGroundingScorer()
    with pytest.raises(ValueError, match="top_k"):
        IndexBackedVisualGroundingScorer(bm25=_BM25(), top_k=41)


def test_index_grounding_ignores_non_identifier_visual_hints():
    bm25 = _BM25()
    scorer = IndexBackedVisualGroundingScorer(bm25=bm25)

    result = scorer(
        {"name": "request flow", "grounding_candidates": ["not/a/symbol"]},
        {"path": "src/runtime.py", "symbol": "compile_repository"},
    )

    assert result is None
    assert bm25.calls == []


def test_index_grounding_ignores_generic_metadata_entities():
    bm25 = _BM25()
    scorer = IndexBackedVisualGroundingScorer(bm25=bm25)

    result = scorer(
        {"name": "preview", "confidence": 0.5},
        {"path": "src/runtime.py", "symbol": "compile_repository"},
    )

    assert result is None
    assert bm25.calls == []


def test_index_grounding_treats_missing_lsp_symbols_as_non_matches():
    class MissingLSP:
        @staticmethod
        def definition(**_arguments):
            raise ValueError("symbol not found")

        @staticmethod
        def references(**_arguments):
            raise ValueError("symbol not found")

    scorer = IndexBackedVisualGroundingScorer(bm25=_BM25(), lsp_provider=MissingLSP())

    result = scorer(
        _entity(),
        {"path": "src/compiler.py", "symbol": "IndexCompiler"},
    )

    assert result["score"] == 0.9
    assert result["evidence"].startswith("BM25 exact definition")


def test_index_grounding_surfaces_backend_failures():
    class FailedLSP:
        @staticmethod
        def definition(**_arguments):
            raise RuntimeError("snapshot changed")

    scorer = IndexBackedVisualGroundingScorer(lsp_provider=FailedLSP())

    with pytest.raises(RuntimeError, match="snapshot changed"):
        scorer(
            _entity(),
            {"path": "src/compiler.py", "symbol": "IndexCompiler"},
        )


def test_index_grounding_prioritizes_index_targets_within_candidate_budget():
    scorer = IndexBackedVisualGroundingScorer(bm25=_BM25(), lsp_provider=_LSP())
    facts = {"facts": [{"entities": [_entity()]}]}
    lexical = [{"path": "src/early.py", "symbol": "Early", "kind": "symbol", "line": 1}]

    candidates = scorer.augment_source_candidates(
        facts,
        lexical,
        max_candidates=2,
    )

    assert candidates == [
        {
            "path": "src/compiler.py",
            "symbol": "IndexCompiler",
            "kind": "symbol",
            "line": 21,
        },
        {
            "path": "src/compiler.py",
            "symbol": "",
            "kind": "source",
            "line": 0,
        },
    ]


def test_index_grounding_candidate_augmentation_respects_zero_budget():
    scorer = IndexBackedVisualGroundingScorer(bm25=_BM25())

    assert (
        scorer.augment_source_candidates(
            {"facts": [{"entities": [_entity()]}]},
            (),
            max_candidates=0,
        )
        == []
    )


def test_index_grounding_queries_bare_qualified_identifier():
    bm25 = _BM25()
    scorer = IndexBackedVisualGroundingScorer(bm25=bm25)

    result = scorer(
        {"name": "src/compiler.py:IndexCompiler", "confidence": 0.9},
        {"path": "src/compiler.py", "symbol": "IndexCompiler"},
    )

    assert result["score"] == 0.9
    assert bm25.calls[0][1]["query"] == "IndexCompiler"
