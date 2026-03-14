"""
Direct tests for embedding_search skill.

Two levels:
- TestEmbeddingSearchSkill: mock vector store, verify skill loading and execution
- TestEmbeddingSearchE2E: real FAISS index, full round-trip.
  Marked @pytest.mark.slow, run with: pytest -v -m slow
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codeminer.agent.routing_agent import SkillRoutingAgent
from codeminer.agent.skills.loader import SkillLoader
from codeminer.agent.skills.registry import SkillRegistry
from codeminer.llm.litellm_chat import LiteLLMChat

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SKILLS_DIR = str(_PROJECT_ROOT / "codeminer" / "agent" / "skills")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    SkillRegistry.reset()
    yield
    SkillRegistry.reset()


@pytest.fixture
def mock_store():
    """A mock CodeVectorStore that returns a fake result list."""
    from codeminer.types import NodeInfo

    store = MagicMock()
    fake_result = NodeInfo(
        node_name="fake_fn",
        type="function",
        file="fake/path.py",
        node_id="fake_node_1",
        start_line=1,
        end_line=10,
        score=0.9,
        content="def fake_fn(): pass",
    )
    store.search_with_content.return_value = [fake_result]
    store.search.return_value = [fake_result]
    return store


@pytest.fixture
def loaded_embedding_search(mock_store):
    """Load only embedding_search into the registry with a mock vector store."""
    from codeminer.ops.retrieve import RetrieveContext

    ctx = MagicMock(spec=RetrieveContext)
    ctx.vector_store = mock_store
    ctx.default_level = "l2"
    ctx.masks = {}

    skill_dir = str(_PROJECT_ROOT / "codeminer" / "agent" / "skills" / "embedding_search")
    loader = SkillLoader()
    meta = loader.load_skill(skill_dir, contexts={"retrieve": ctx})
    assert meta is not None
    SkillRegistry().register(meta)
    return meta


# ---------------------------------------------------------------------------
# TestEmbeddingSearchSkill
# ---------------------------------------------------------------------------


class TestEmbeddingSearchSkill:
    """
    Direct skill tests with mock vector store.

    Test that embedding_search skill can be loaded, and its executor
    correctly calls the vector store with the right parameters.
    """

    def test_skill_loads_successfully(self, loaded_embedding_search):
        """embedding_search skill must load into the registry."""
        assert loaded_embedding_search is not None
        assert loaded_embedding_search.skill_id == "embedding_search"

    def test_skill_has_executor(self, loaded_embedding_search):
        """Loaded skill must have a callable executor function."""
        executor_fn = loaded_embedding_search.executor_fn
        assert executor_fn is not None
        assert callable(executor_fn)

    def test_executor_calls_vector_store(self, loaded_embedding_search, mock_store):
        """Executor must call vector store's search_with_content method."""
        executor_fn = loaded_embedding_search.executor_fn

        results = executor_fn(query="test query", top_k=5, return_content=True)

        mock_store.search_with_content.assert_called_once()
        assert isinstance(results, list)

    def test_executor_forwards_query_parameter(self, loaded_embedding_search, mock_store):
        """Query parameter must be passed to vector store."""
        executor_fn = loaded_embedding_search.executor_fn
        query = "find authentication logic"

        executor_fn(query=query, top_k=5, return_content=True)

        call_args = mock_store.search_with_content.call_args
        assert call_args[1]["query"] == query

    def test_executor_forwards_top_k(self, loaded_embedding_search, mock_store):
        """top_k parameter must be passed to vector store."""
        executor_fn = loaded_embedding_search.executor_fn

        executor_fn(query="test", top_k=10, return_content=True)

        call_args = mock_store.search_with_content.call_args
        assert call_args[1]["top_k"] == 10

    def test_executor_uses_search_when_return_content_false(self, loaded_embedding_search, mock_store):
        """When return_content=False, must use search() instead of search_with_content()."""
        executor_fn = loaded_embedding_search.executor_fn

        executor_fn(query="test", top_k=5, return_content=False)

        mock_store.search.assert_called_once()
        mock_store.search_with_content.assert_not_called()

    def test_executor_result_has_expected_fields(self, loaded_embedding_search, mock_store):
        """Results must have score, file, and content fields."""
        executor_fn = loaded_embedding_search.executor_fn

        results = executor_fn(query="test", top_k=5, return_content=True)

        assert len(results) > 0
        node = results[0]
        assert hasattr(node, "score")
        assert hasattr(node, "file")
        assert hasattr(node, "content")

    def test_skill_works_with_compiler(self, loaded_embedding_search):
        """Skill invocation must work with SkillCompiler."""
        from codeminer.compiler.skills.compiler import SkillCompiler

        invocations = [
            {
                "skill_id": "embedding_search",
                "inputs": {
                    "query": "test query",
                    "top_k": 5,
                    "return_content": True,
                },
            }
        ]

        compilation = SkillCompiler().compile(invocations)
        assert len(compilation.execution_plan.stages) > 0
        assert "embedding_search" in compilation.skill_dag.nodes


# ---------------------------------------------------------------------------
# TestEmbeddingSearchE2E
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEmbeddingSearchE2E:
    """
    Full round-trip: real FAISS index executes embedding_search queries.

    Index is cached at /tmp/codeminer_e2e_index (reused across runs).
    Set CODEMINER_INDEX_PATH to override.
    """

    @pytest.fixture(scope="class", autouse=True)
    def reset_for_class(self):
        SkillRegistry.reset()
        yield
        SkillRegistry.reset()

    @pytest.fixture(scope="class")
    def real_embedding_search(self):
        """Load embedding_search skill with real FAISS index."""
        import os
        from codeminer.index.embedding import CodeVectorStore, build_hierarchical_vector_store
        from codeminer.ops.retrieve import RetrieveContext

        # Build or load the FAISS index
        index_path = os.environ.get("CODEMINER_INDEX_PATH", "/tmp/codeminer_e2e_index")
        l0 = Path(index_path) / "l0"
        l2 = Path(index_path) / "l2"

        if l0.exists() and l2.exists():
            print(f"Loading existing index from {index_path}")
            vs = CodeVectorStore(
                embedding_model="nomic-ai/CodeRankEmbed",
                embedding_provider="huggingface",
                dimension=768,
                store_path=index_path,
                model_kwargs={"trust_remote_code": True},
                encode_kwargs={
                    "batch_size": 8,  # Reduce batch size to avoid OOM
                    "normalize_embeddings": True,
                },
            )
            vs.load(index_path)
        else:
            print(f"Building new index at {index_path}")
            Path(index_path).mkdir(parents=True, exist_ok=True)
            vs = build_hierarchical_vector_store(
                repo_path=str(_PROJECT_ROOT),
                index_path=index_path,
                languages=["python"],
                embedding_model="nomic-ai/CodeRankEmbed",
                embedding_provider="huggingface",
                embedding_dimension=768,
                embedding_kwargs={
                    "model_kwargs": {"trust_remote_code": True},
                    "encode_kwargs": {
                        "batch_size": 8,  # Reduce batch size to avoid OOM
                        "normalize_embeddings": True,
                    },
                },
            )

        ctx = RetrieveContext(vector_store=vs, default_level="l2")
        loader = SkillLoader()

        # Load only embedding_search skill
        skill_dir = str(_PROJECT_ROOT / "codeminer" / "agent" / "skills" / "embedding_search")
        meta = loader.load_skill(skill_dir, contexts={"retrieve": ctx})
        assert meta is not None
        SkillRegistry().register(meta)

        return meta

    def test_embedding_search_returns_results(self, real_embedding_search):
        """embedding_search must return results for a semantic query."""
        executor_fn = real_embedding_search.executor_fn

        results = executor_fn(
            query="function that performs semantic similarity search over code vectors",
            top_k=5,
            return_content=True,
        )

        assert isinstance(results, list)
        assert len(results) > 0
        print(f"Got {len(results)} results")

    def test_results_are_relevant(self, real_embedding_search):
        """Top results for a code-search query must mention relevant file paths."""
        executor_fn = real_embedding_search.executor_fn

        results = executor_fn(
            query="function that performs semantic similarity search over code vectors",
            top_k=5,
            return_content=True,
        )

        files = [r.file for r in results]
        print(f"Result files: {files}")
        relevant = [f for f in files if "vector" in f or "embedding" in f or "search" in f]
        assert relevant, f"No relevant files in top results: {files}"

    def test_results_have_scores(self, real_embedding_search):
        """Results must have similarity scores."""
        executor_fn = real_embedding_search.executor_fn

        results = executor_fn(
            query="vector store implementation",
            top_k=3,
            return_content=True,
        )

        assert len(results) > 0
        for result in results:
            assert hasattr(result, "score")
            assert result.score is not None
            # Inner product scores can be any positive value (not normalized to [0,1])
            # Higher scores indicate better matches
            assert result.score > 0, f"Score {result.score} should be positive"

        # Verify scores are in descending order (best match first)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Scores should be in descending order"

    def test_different_levels(self, real_embedding_search):
        """Test searching at different hierarchy levels (l0 vs l2)."""
        executor_fn = real_embedding_search.executor_fn

        # Search at l0 (file-level skeletons)
        l0_results = executor_fn(
            query="embedding implementation",
            top_k=3,
            level="l0",
            return_content=True,
        )

        # Search at l2 (function/method level)
        l2_results = executor_fn(
            query="embedding implementation",
            top_k=3,
            level="l2",
            return_content=True,
        )

        assert len(l0_results) > 0
        assert len(l2_results) > 0
        print(f"L0 results: {len(l0_results)}, L2 results: {len(l2_results)}")

    @pytest.mark.parametrize(
        "vertex_model",
        [
            # Three chat models provided via Vertex AI in this project.
            "vertex_ai/gemini-2.5-flash",
            "vertex_ai/gemini-2.0-flash",
            "vertex_ai/claude-sonnet-4@20250514",
        ],
    )
    def test_vertex_routing_selects_embedding_search(
        self, real_embedding_search, vertex_model
    ):
        """
        Use Vertex-provided chat models as the routing LLM and verify that
        they select the embedding_search skill for a semantic code-search query.
        """
        import os

        # Allow overriding the model name via environment, default to the parametrized value.
        model_name = os.environ.get("CODEMINER_VERTEX_ROUTING_MODEL", vertex_model)

        llm = LiteLLMChat(
            model=model_name,
            temperature=0.0,
            max_tokens=512,
        )
        agent = SkillRoutingAgent(llm=llm)

        try:
            invocations = agent.route(
                "function that performs semantic similarity search over code vectors",
                top_k=5,
            )
        except Exception as exc:
            # If Vertex AI / LiteLLM is not configured correctly, skip instead of failing.
            pytest.skip(f"Vertex routing unavailable for {model_name}: {exc}")

        assert isinstance(invocations, list)
        assert invocations, f"Expected at least one skill invocation for {model_name}"

        skill_ids = [inv.get("skill_id") for inv in invocations]
        assert (
            "embedding_search" in skill_ids
        ), f"Expected embedding_search for {model_name}, got: {skill_ids}"