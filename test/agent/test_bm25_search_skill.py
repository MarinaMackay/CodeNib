"""
Unit and integration tests for the bm25_search skill.

Structure:
- TestBM25SearchExecutor: pure unit tests using mock RetrieveContext,
  no real index or network required.
- TestBM25SearchLoader: integration tests that load the real skill
  package directory through SkillLoader, still using a mock context.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from codeminer.agent.skills.core import Cost, SkillType
from codeminer.agent.skills.loader import SkillLoader
from codeminer.agent.skills.registry import SkillRegistry

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SKILL_DIR = str(_PROJECT_ROOT / "codeminer" / "agent" / "skills" / "bm25_search")


def _make_context(*, bm25: Any = None) -> MagicMock:
    """Build a lightweight mock that quacks like RetrieveContext."""
    ctx = MagicMock()
    ctx.bm25 = bm25
    return ctx


def _make_bm25() -> MagicMock:
    """Build a mock BM25CodeIndexer with a search method."""
    bm25 = MagicMock()
    bm25.search.return_value = []
    return bm25


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure the SkillRegistry singleton is clean before and after each test."""
    SkillRegistry.reset()
    yield
    SkillRegistry.reset()


class TestBM25SearchExecutor:
    """Tests that target executor.py directly via create_executor()."""

    def _load_executor(self, context: Any):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "bm25_search.executor",
            os.path.join(SKILL_DIR, "executor.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.create_executor(context)

    def test_raises_when_bm25_is_none(self):
        """execute() must raise RuntimeError if context.bm25 is None."""
        ctx = _make_context(bm25=None)
        execute = self._load_executor(ctx)
        with pytest.raises(RuntimeError, match="BM25 index not available"):
            execute("any query")

    def test_default_call_invokes_search(self):
        """A basic call must invoke bm25.search() exactly once."""
        bm25 = _make_bm25()
        ctx = _make_context(bm25=bm25)
        execute = self._load_executor(ctx)

        execute("some query")

        bm25.search.assert_called_once()

    def test_top_k_is_forwarded(self):
        """top_k parameter must be passed through to bm25.search()."""
        bm25 = _make_bm25()
        ctx = _make_context(bm25=bm25)
        execute = self._load_executor(ctx)

        execute("query", top_k=7)

        _, kwargs = bm25.search.call_args
        assert kwargs["top_k"] == 7

    def test_default_top_k(self):
        """When top_k is not passed, default (20) must be used."""
        bm25 = _make_bm25()
        ctx = _make_context(bm25=bm25)
        execute = self._load_executor(ctx)

        execute("query")

        _, kwargs = bm25.search.call_args
        assert kwargs["top_k"] == 20

    def test_filter_test_forwarded(self):
        """filter_test kwarg must be forwarded to bm25.search()."""
        bm25 = _make_bm25()
        ctx = _make_context(bm25=bm25)
        execute = self._load_executor(ctx)

        execute("query", filter_test=True)

        _, kwargs = bm25.search.call_args
        assert kwargs["filter_test"] is True

    def test_filter_test_defaults_to_false(self):
        """When filter_test is not given, False must be forwarded."""
        bm25 = _make_bm25()
        ctx = _make_context(bm25=bm25)
        execute = self._load_executor(ctx)

        execute("query")

        _, kwargs = bm25.search.call_args
        assert kwargs["filter_test"] is False

    def test_return_content_forwarded(self):
        """return_content kwarg must be forwarded as return_code_content."""
        bm25 = _make_bm25()
        ctx = _make_context(bm25=bm25)
        execute = self._load_executor(ctx)

        execute("query", return_content=False)

        _, kwargs = bm25.search.call_args
        assert kwargs["return_code_content"] is False

    def test_return_content_defaults_to_true(self):
        """When return_content is not given, True must be forwarded."""
        bm25 = _make_bm25()
        ctx = _make_context(bm25=bm25)
        execute = self._load_executor(ctx)

        execute("query")

        _, kwargs = bm25.search.call_args
        assert kwargs["return_code_content"] is True

    def test_wrap_with_line_numbers_forwarded(self):
        """wrap_with_line_numbers kwarg must be forwarded as wrap_with_ln."""
        bm25 = _make_bm25()
        ctx = _make_context(bm25=bm25)
        execute = self._load_executor(ctx)

        execute("query", wrap_with_line_numbers=False)

        _, kwargs = bm25.search.call_args
        assert kwargs["wrap_with_ln"] is False

    def test_wrap_with_line_numbers_defaults_to_true(self):
        """When wrap_with_line_numbers is not given, True must be forwarded."""
        bm25 = _make_bm25()
        ctx = _make_context(bm25=bm25)
        execute = self._load_executor(ctx)

        execute("query")

        _, kwargs = bm25.search.call_args
        assert kwargs["wrap_with_ln"] is True

    def test_returns_bm25_results_unchanged(self):
        """execute() must return whatever bm25.search() returns."""
        bm25 = _make_bm25()
        fake_results = [MagicMock(), MagicMock()]
        bm25.search.return_value = fake_results
        ctx = _make_context(bm25=bm25)
        execute = self._load_executor(ctx)

        result = execute("query")

        assert result is fake_results

    def test_query_is_forwarded(self):
        """The query string must be passed to bm25.search()."""
        bm25 = _make_bm25()
        ctx = _make_context(bm25=bm25)
        execute = self._load_executor(ctx)

        execute("find function foo")

        _, kwargs = bm25.search.call_args
        assert kwargs["query"] == "find function foo"


class TestBM25SearchLoader:
    """
    Tests that load the actual bm25_search skill directory.
    No real BM25 index is needed; a mock context is enough.
    """

    @pytest.fixture
    def loaded_meta(self):
        """Load the real bm25_search skill with a mock RetrieveContext."""
        mock_ctx = _make_context(bm25=_make_bm25())
        loader = SkillLoader()
        meta = loader.load_skill(SKILL_DIR, contexts={"retrieve": mock_ctx})
        assert meta is not None, f"SkillLoader returned None for {SKILL_DIR}"
        SkillRegistry().register(meta)
        return meta

    def test_skill_id(self, loaded_meta):
        assert loaded_meta.skill_id == "bm25_search"

    def test_skill_type_is_retrieval(self, loaded_meta):
        assert loaded_meta.skill_type == SkillType.RETRIEVAL

    def test_cost_is_low(self, loaded_meta):
        assert loaded_meta.cost == Cost.LOW

    def test_query_input_is_required(self, loaded_meta):
        query_input = next(
            (i for i in loaded_meta.inputs if i.name == "query"), None
        )
        assert query_input is not None, "query input spec not found"
        assert query_input.required is True
        assert query_input.type_hint == "str"

    def test_optional_inputs_present(self, loaded_meta):
        """top_k and filter_test must both exist."""
        names = {i.name for i in loaded_meta.inputs}
        expected = {"top_k", "filter_test"}
        missing = expected - names
        assert not missing, f"Missing optional inputs: {missing}"

    def test_optional_inputs_not_required(self, loaded_meta):
        optional_names = {"top_k", "filter_test"}
        for spec in loaded_meta.inputs:
            if spec.name in optional_names:
                assert spec.required is False, (
                    f"Input '{spec.name}' should be optional but required=True"
                )

    def test_output_type(self, loaded_meta):
        assert "QueriedNode" in loaded_meta.outputs.type_hint

    def test_index_requirement_is_bm25(self, loaded_meta):
        assert len(loaded_meta.index_requirements) >= 1
        types = {r.index_type for r in loaded_meta.index_requirements}
        assert "bm25" in types

    def test_resources_empty(self, loaded_meta):
        """BM25 does not require GPU or other special resources."""
        assert loaded_meta.resources == []

    def test_skill_doc_populated(self, loaded_meta):
        """skill.md content must be loaded into skill_doc."""
        assert len(loaded_meta.skill_doc) > 0
        assert "bm25" in loaded_meta.skill_doc.lower()

    def test_executor_fn_is_not_none(self, loaded_meta):
        assert loaded_meta.executor_fn is not None

    def test_executor_fn_is_callable(self, loaded_meta):
        assert callable(loaded_meta.executor_fn)

    def test_executor_fn_returns_list(self, loaded_meta):
        """Calling executor_fn with a query must return a list (possibly empty)."""
        result = loaded_meta.executor_fn("test query", top_k=3)
        assert isinstance(result, list)

    def test_defaults_top_k(self, loaded_meta):
        assert loaded_meta.defaults.get("top_k") == 20

    def test_defaults_filter_test(self, loaded_meta):
        assert loaded_meta.defaults.get("filter_test") is False

    def test_defaults_return_content(self, loaded_meta):
        assert loaded_meta.defaults.get("return_content") is True

    def test_defaults_wrap_with_line_numbers(self, loaded_meta):
        assert loaded_meta.defaults.get("wrap_with_line_numbers") is True

    def test_registered_in_registry(self, loaded_meta):
        reg = SkillRegistry()
        assert reg.has("bm25_search")

    def test_metadata_serializable(self, loaded_meta):
        """to_dict() must produce a JSON-serializable dict."""
        import json

        d = loaded_meta.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["skill_id"] == "bm25_search"
        assert parsed["cost"] == "low"
