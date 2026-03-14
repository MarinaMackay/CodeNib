"""
Unit and integration tests for the embedding_search skill.

Structure:
- TestEmbeddingSearchExecutor: pure unit tests using mock RetrieveContext,
  no real index or network required.
- TestEmbeddingSearchLoader: integration tests that load the real skill
  package directory through SkillLoader, still using a mock context.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from codeminer.agent.skills.core import Cost, SkillType
from codeminer.agent.skills.loader import SkillLoader
from codeminer.agent.skills.registry import SkillRegistry

# Path to the real embedding_search package on disk.
# Resolved relative to this file so tests work regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SKILL_DIR = str(_PROJECT_ROOT / "codeminer" / "agent" / "skills" / "embedding_search")



def _make_context(
    *,
    vector_store: Any = None,
    default_level: str = "l2",
    masks: dict | None = None,
) -> MagicMock:
    """Build a lightweight mock that quacks like RetrieveContext."""
    ctx = MagicMock()
    ctx.vector_store = vector_store
    ctx.default_level = default_level
    ctx.masks = masks if masks is not None else {}
    return ctx


def _make_store() -> MagicMock:
    """Build a mock CodeVectorStore with search methods."""
    store = MagicMock()
    store.search.return_value = []
    store.search_with_content.return_value = []
    return store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure the SkillRegistry singleton is clean before and after each test."""
    SkillRegistry.reset()
    yield
    SkillRegistry.reset()



class TestEmbeddingSearchExecutor:
    """Tests that target executor.py directly via create_executor()."""

    def _load_executor(self, context: Any):
        """Import create_executor from the real executor module and call it."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "embedding_search.executor",
            os.path.join(SKILL_DIR, "executor.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.create_executor(context)

    def test_raises_when_vector_store_is_none(self):
        """execute() must raise RuntimeError if context.vector_store is None."""
        ctx = _make_context(vector_store=None)
        execute = self._load_executor(ctx)
        with pytest.raises(RuntimeError, match="Vector store not available"):
            execute("any query")

    def test_return_content_true_calls_search_with_content(self):
        """When return_content=True, search_with_content() should be called."""
        store = _make_store()
        ctx = _make_context(vector_store=store)
        execute = self._load_executor(ctx)

        execute("some query", top_k=5, return_content=True)

        store.search_with_content.assert_called_once()
        store.search.assert_not_called()

    def test_return_content_false_calls_search(self):
        """When return_content=False, search() should be called instead."""
        store = _make_store()
        ctx = _make_context(vector_store=store)
        execute = self._load_executor(ctx)

        execute("some query", top_k=5, return_content=False)

        store.search.assert_called_once()
        store.search_with_content.assert_not_called()

    def test_top_k_is_forwarded(self):
        """top_k parameter must be passed through to the store call."""
        store = _make_store()
        ctx = _make_context(vector_store=store)
        execute = self._load_executor(ctx)

        execute("query", top_k=7, return_content=True)

        _, kwargs = store.search_with_content.call_args
        assert kwargs["top_k"] == 7

    def test_level_kwarg_overrides_context_default(self):
        """Explicit level= kwarg must take precedence over context.default_level."""
        store = _make_store()
        ctx = _make_context(vector_store=store, default_level="l2")
        execute = self._load_executor(ctx)

        execute("query", level="l0", return_content=True)

        _, kwargs = store.search_with_content.call_args
        assert kwargs["level"] == "l0"

    def test_level_falls_back_to_context_default(self):
        """When level is not passed, context.default_level must be used."""
        store = _make_store()
        ctx = _make_context(vector_store=store, default_level="l0")
        execute = self._load_executor(ctx)

        execute("query", return_content=True)

        _, kwargs = store.search_with_content.call_args
        assert kwargs["level"] == "l0"

    def test_score_threshold_forwarded(self):
        """score_threshold must be forwarded to the store call."""
        store = _make_store()
        ctx = _make_context(vector_store=store)
        execute = self._load_executor(ctx)

        execute("query", score_threshold=0.75, return_content=True)

        _, kwargs = store.search_with_content.call_args
        assert kwargs["score_threshold"] == 0.75

    def test_score_threshold_defaults_to_none(self):
        """When score_threshold is not given, None must be forwarded."""
        store = _make_store()
        ctx = _make_context(vector_store=store)
        execute = self._load_executor(ctx)

        execute("query", return_content=False)

        _, kwargs = store.search.call_args
        assert kwargs["score_threshold"] is None

    def test_mask_name_resolves_ids_from_context(self):
        """mask_name must be looked up in context.masks and forwarded as mask_ids."""
        store = _make_store()
        mask_ids = {"node_1", "node_2"}
        ctx = _make_context(vector_store=store, masks={"my_mask": mask_ids})
        execute = self._load_executor(ctx)

        execute("query", mask_name="my_mask", return_content=False)

        _, kwargs = store.search.call_args
        assert kwargs["mask_node_ids"] == mask_ids

    def test_unknown_mask_name_gives_none(self):
        """An unrecognized mask_name must produce mask_node_ids=None, not an error."""
        store = _make_store()
        ctx = _make_context(vector_store=store, masks={})
        execute = self._load_executor(ctx)

        execute("query", mask_name="nonexistent", return_content=False)

        _, kwargs = store.search.call_args
        assert kwargs["mask_node_ids"] is None

    def test_no_mask_name_gives_none(self):
        """When mask_name is not supplied, mask_node_ids must be None."""
        store = _make_store()
        ctx = _make_context(vector_store=store)
        execute = self._load_executor(ctx)

        execute("query", return_content=False)

        _, kwargs = store.search.call_args
        assert kwargs["mask_node_ids"] is None

    def test_returns_store_results_unchanged(self):
        """execute() must return whatever the store method returns."""
        store = _make_store()
        fake_results = [MagicMock(), MagicMock()]
        store.search_with_content.return_value = fake_results
        ctx = _make_context(vector_store=store)
        execute = self._load_executor(ctx)

        result = execute("query", return_content=True)

        assert result is fake_results


class TestEmbeddingSearchLoader:
    """
    Tests that load the actual embedding_search skill directory.
    The FAISS index is not needed here; a mock context is enough.
    """

    @pytest.fixture
    def loaded_meta(self):
        """Load the real embedding_search skill with a mock RetrieveContext.

        load_skill() parses and returns metadata but does not register it.
        We register manually here so that tests covering registry behaviour
        see the same state as production code (which goes through load_all()).
        """
        mock_ctx = _make_context(vector_store=_make_store())
        loader = SkillLoader()
        meta = loader.load_skill(SKILL_DIR, contexts={"retrieve": mock_ctx})
        assert meta is not None, f"SkillLoader returned None for {SKILL_DIR}"
        SkillRegistry().register(meta)
        return meta

    def test_skill_id(self, loaded_meta):
        assert loaded_meta.skill_id == "embedding_search"

    def test_skill_type_is_retrieval(self, loaded_meta):
        assert loaded_meta.skill_type == SkillType.RETRIEVAL

    def test_cost_is_high(self, loaded_meta):
        assert loaded_meta.cost == Cost.HIGH

    def test_query_input_is_required(self, loaded_meta):
        query_input = next(
            (i for i in loaded_meta.inputs if i.name == "query"), None
        )
        assert query_input is not None, "query input spec not found"
        assert query_input.required is True
        assert query_input.type_hint == "str"

    def test_optional_inputs_present(self, loaded_meta):
        """top_k, level, return_content, score_threshold, mask_name must all exist."""
        names = {i.name for i in loaded_meta.inputs}
        expected = {"top_k", "level", "return_content", "score_threshold", "mask_name"}
        missing = expected - names
        assert not missing, f"Missing optional inputs: {missing}"

    def test_optional_inputs_not_required(self, loaded_meta):
        optional_names = {
            "top_k", "level", "return_content", "score_threshold", "mask_name"
        }
        for spec in loaded_meta.inputs:
            if spec.name in optional_names:
                assert spec.required is False, (
                    f"Input '{spec.name}' should be optional but required=True"
                )

    def test_output_type(self, loaded_meta):
        assert "QueriedNode" in loaded_meta.outputs.type_hint

    def test_index_requirement_is_vector(self, loaded_meta):
        assert len(loaded_meta.index_requirements) >= 1
        types = {r.index_type for r in loaded_meta.index_requirements}
        assert "vector" in types

    def test_resources_include_gpu(self, loaded_meta):
        assert "GPU" in loaded_meta.resources

    def test_skill_doc_populated(self, loaded_meta):
        """skill.md content must be loaded into skill_doc."""
        assert len(loaded_meta.skill_doc) > 0
        assert "embedding" in loaded_meta.skill_doc.lower()

    def test_executor_fn_is_not_none(self, loaded_meta):
        """A mock context must be enough to produce a callable executor."""
        assert loaded_meta.executor_fn is not None

    def test_executor_fn_is_callable(self, loaded_meta):
        assert callable(loaded_meta.executor_fn)

    def test_executor_fn_returns_list(self, loaded_meta):
        """Calling executor_fn with a query must return a list (possibly empty)."""
        result = loaded_meta.executor_fn("test query", top_k=3)
        assert isinstance(result, list)

    def test_defaults_top_k(self, loaded_meta):
        assert loaded_meta.defaults.get("top_k") == 20

    def test_defaults_level(self, loaded_meta):
        assert loaded_meta.defaults.get("level") == "l2"

    def test_defaults_return_content(self, loaded_meta):
        assert loaded_meta.defaults.get("return_content") is True

    def test_registered_in_registry(self, loaded_meta):
        """Loading a skill must automatically register it in the singleton."""
        reg = SkillRegistry()
        assert reg.has("embedding_search")

    def test_metadata_serializable(self, loaded_meta):
        """to_dict() must produce a JSON-serializable dict."""
        import json

        d = loaded_meta.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["skill_id"] == "embedding_search"
        assert parsed["cost"] == "high"
