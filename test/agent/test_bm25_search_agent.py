"""
Layer 3 tests: can the routing agent correctly select and invoke bm25_search?

Two levels:
- TestAgentSelectsBM25Search : mock LLM, verify routing logic wires up
  bm25_search and the executor is actually called with the right inputs.
- TestAgentCallsBM25SearchE2E_*: real LLM + real BM25 index, full round-trip.
  Three model variants: gemini-2.0-flash, gemini-2.5-flash, claude-sonnet-4.
  Marked @pytest.mark.slow, run with: pytest -v -m slow
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codeminer.agent.routing_agent import RoutingPlan, SkillInvocationSchema, SkillRoutingAgent
from codeminer.agent.skills.loader import SkillLoader
from codeminer.agent.skills.registry import SkillRegistry
from codeminer.llm.litellm_chat import LiteLLMChat

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SKILLS_DIR = str(_PROJECT_ROOT / "codeminer" / "agent" / "skills")
_VERTEX_PROJECT = "ucsd-cse-stable-gcp"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    SkillRegistry.reset()
    yield
    SkillRegistry.reset()


@pytest.fixture
def mock_bm25():
    """A mock BM25CodeIndexer that returns a fake result list."""
    from codeminer.types import NodeInfo

    bm25 = MagicMock()
    fake_result = NodeInfo(
        node_name="fake_indexer",
        type="class",
        file="fake/bm25.py",
        node_id="fake_bm25_node",
        start_line=1,
        end_line=20,
        score=0.0,
        content="class FakeIndexer: pass",
    )
    bm25.search.return_value = [fake_result]
    return bm25


@pytest.fixture
def loaded_bm25_search(mock_bm25):
    """Load only bm25_search into the registry with a mock BM25 indexer."""
    from codeminer.ops.retrieve import RetrieveContext

    ctx = MagicMock(spec=RetrieveContext)
    ctx.bm25 = mock_bm25

    skill_dir = str(_PROJECT_ROOT / "codeminer" / "agent" / "skills" / "bm25_search")
    loader = SkillLoader()
    meta = loader.load_skill(skill_dir, contexts={"retrieve": ctx})
    assert meta is not None
    SkillRegistry().register(meta)
    return meta


def _make_llm_returning(plan: RoutingPlan) -> MagicMock:
    """Build a mock LLM that returns a fixed RoutingPlan."""
    llm = MagicMock(spec=LiteLLMChat)
    structured = MagicMock()
    structured.invoke.return_value = plan
    llm.with_structured_output.return_value = structured
    return llm


# ---------------------------------------------------------------------------
# TestAgentSelectsBM25Search (mock LLM, 3 tests)
# ---------------------------------------------------------------------------


class TestAgentSelectsBM25Search:
    """
    Mock LLM tests for bm25_search.

    The LLM always returns a plan that picks bm25_search.
    We verify routing, executor invocation, result fields, and compiler
    compatibility.
    """

    def _plan_for_bm25_search(self, query: str, top_k: int = 5) -> RoutingPlan:
        return RoutingPlan(
            reasoning="keyword query, use BM25 search",
            invocations=[
                SkillInvocationSchema(
                    skill_id="bm25_search",
                    inputs={"query": query, "top_k": top_k},
                )
            ],
        )

    def test_route_and_execute(self, loaded_bm25_search, mock_bm25):
        """route() selects bm25_search, forwards correct query, and executor
        calls the underlying BM25 indexer."""
        query = "HTTPClient"
        llm = _make_llm_returning(self._plan_for_bm25_search(query, top_k=3))
        agent = SkillRoutingAgent(llm=llm)

        invocations = agent.route(query, top_k=3)

        skill_ids = [inv["skill_id"] for inv in invocations]
        assert "bm25_search" in skill_ids

        bm25_inv = next(i for i in invocations if i["skill_id"] == "bm25_search")
        assert bm25_inv["inputs"]["query"] == query

        executor_fn = SkillRegistry().get("bm25_search").executor_fn
        results = executor_fn(**bm25_inv["inputs"])

        mock_bm25.search.assert_called_once()
        assert isinstance(results, list)
        assert len(results) > 0

    def test_executor_result_has_expected_fields(self, loaded_bm25_search, mock_bm25):
        """The executor output must have node_name, type, and file fields."""
        query = "CodeGraph"
        llm = _make_llm_returning(self._plan_for_bm25_search(query))
        agent = SkillRoutingAgent(llm=llm)

        invocations = agent.route(query, top_k=5)
        bm25_inv = next(i for i in invocations if i["skill_id"] == "bm25_search")
        executor_fn = SkillRegistry().get("bm25_search").executor_fn

        results = executor_fn(**bm25_inv["inputs"])

        node = results[0]
        assert hasattr(node, "node_name")
        assert hasattr(node, "type")
        assert hasattr(node, "file")

    def test_invocation_is_accepted_by_skill_compiler(self, loaded_bm25_search):
        """The invocations list from route() must pass through SkillCompiler.compile()."""
        from codeminer.compiler.skills.compiler import SkillCompiler

        llm = _make_llm_returning(self._plan_for_bm25_search("TODO"))
        agent = SkillRoutingAgent(llm=llm)

        invocations = agent.route("TODO", top_k=5)

        compilation = SkillCompiler().compile(invocations)
        assert len(compilation.execution_plan.stages) > 0
        assert "bm25_search" in compilation.skill_dag.nodes


# ---------------------------------------------------------------------------
# E2E helpers
# ---------------------------------------------------------------------------


def _build_or_load_bm25_index():
    """Build (or load cached) BM25 index and return (indexer, registry, loader)."""
    from codeminer.code_chunker import CodeChunker, RepoChunkingConfig
    from codeminer.index.sparse_idx.bm25_index import BM25CodeIndexer
    from codeminer.ops.retrieve import RetrieveContext

    index_path = os.environ.get(
        "CODEMINER_BM25_INDEX_PATH", "/tmp/codeminer_bm25_e2e_index"
    )
    documents_file = Path(index_path) / "documents.json"

    if documents_file.exists():
        indexer = BM25CodeIndexer(max_k=128)
        indexer.load_index(index_path)
    else:
        chunker = CodeChunker(
            language="python",
            repo_config=RepoChunkingConfig(languages=["python"]),
            max_lines_per_chunk=300,
        )
        chunks = chunker.chunk_repository(repo_path=str(_PROJECT_ROOT))
        assert chunks, "No code chunks generated"
        indexer = BM25CodeIndexer(chunks=chunks, max_k=128)
        indexer.project_root = str(_PROJECT_ROOT)
        Path(index_path).mkdir(parents=True, exist_ok=True)
        indexer.save_index(index_path)

    registry = SkillRegistry.__new__(SkillRegistry)
    registry._skills = {}

    ctx = RetrieveContext(bm25=indexer)
    loader = SkillLoader()
    loader.load_all(SKILLS_DIR, contexts={"retrieve": ctx}, registry=registry)

    return registry


def _make_e2e_agent(
    model: str, registry: SkillRegistry, vertex_location: str = "us-central1"
) -> SkillRoutingAgent:
    """Create a SkillRoutingAgent backed by a real LLM."""
    llm = LiteLLMChat(
        model=model,
        temperature=0.0,
        max_tokens=1024,
        extra_kwargs={
            "vertex_project": _VERTEX_PROJECT,
            "vertex_location": vertex_location,
        },
    )
    return SkillRoutingAgent(llm=llm, registry=registry)


# ---------------------------------------------------------------------------
# TestAgentCallsBM25SearchE2E -- gemini-2.0-flash (3 tests)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAgentCallsBM25SearchE2E_Gemini20Flash:
    """
    Full round-trip with vertex_ai/gemini-2.0-flash.
    BM25 index is cached at /tmp/codeminer_bm25_e2e_index.
    """

    @pytest.fixture(autouse=True)
    def clean_registry(self):
        yield

    @pytest.fixture(scope="class")
    def bm25_env(self):
        registry = _build_or_load_bm25_index()
        agent = _make_e2e_agent("vertex_ai/gemini-2.0-flash", registry)
        return agent, registry

    def test_keyword_query_routes_to_bm25_search(self, bm25_env):
        """An exact-keyword query must cause the agent to select bm25_search."""
        agent, _ = bm25_env
        result = agent.route("BM25CodeIndexer", top_k=5)
        skill_ids = [inv["skill_id"] for inv in result]
        assert "bm25_search" in skill_ids, (
            f"Expected bm25_search for keyword query, got: {skill_ids}"
        )

    def test_bm25_search_returns_results(self, bm25_env):
        """Running the invocations must produce a non-empty result list."""
        agent, registry = bm25_env
        invocations = agent.route("BM25CodeIndexer", top_k=5)
        bm25_inv = next(
            (i for i in invocations if i["skill_id"] == "bm25_search"), None
        )
        assert bm25_inv is not None

        executor_fn = registry.get("bm25_search").executor_fn
        results = executor_fn(**bm25_inv["inputs"])

        assert isinstance(results, list)
        assert len(results) > 0

    def test_results_are_relevant(self, bm25_env):
        """Top results for a keyword query must reference relevant nodes."""
        agent, registry = bm25_env
        invocations = agent.route("BM25CodeIndexer", top_k=10)
        bm25_inv = next(i for i in invocations if i["skill_id"] == "bm25_search")
        executor_fn = registry.get("bm25_search").executor_fn
        results = executor_fn(**bm25_inv["inputs"])

        names = [r.node_name for r in results]
        relevant = [n for n in names if "bm25" in n.lower()]
        assert relevant, f"No BM25-related nodes in top results: {names}"


# ---------------------------------------------------------------------------
# TestAgentCallsBM25SearchE2E -- gemini-2.5-flash 
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAgentCallsBM25SearchE2E_Gemini25Flash:
    """Full round-trip with vertex_ai/gemini-2.5-flash."""

    @pytest.fixture(autouse=True)
    def clean_registry(self):
        yield

    @pytest.fixture(scope="class")
    def bm25_env(self):
        registry = _build_or_load_bm25_index()
        agent = _make_e2e_agent("vertex_ai/gemini-2.5-flash", registry)
        return agent, registry

    def test_routes_and_returns_relevant_results(self, bm25_env):
        """gemini-2.5-flash must route to bm25_search, execute, and return
        relevant results for a keyword query."""
        agent, registry = bm25_env
        invocations = agent.route("BM25CodeIndexer", top_k=10)

        skill_ids = [inv["skill_id"] for inv in invocations]
        assert "bm25_search" in skill_ids, (
            f"Expected bm25_search, got: {skill_ids}"
        )

        bm25_inv = next(i for i in invocations if i["skill_id"] == "bm25_search")
        executor_fn = registry.get("bm25_search").executor_fn
        results = executor_fn(**bm25_inv["inputs"])

        assert isinstance(results, list) and len(results) > 0
        names = [r.node_name for r in results]
        relevant = [n for n in names if "bm25" in n.lower()]
        assert relevant, f"No BM25-related nodes in top results: {names}"


# ---------------------------------------------------------------------------
# TestAgentCallsBM25SearchE2E -- claude-sonnet-4
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAgentCallsBM25SearchE2E_ClaudeSonnet4:
    """Full round-trip with vertex_ai/claude-sonnet-4@20250514."""

    @pytest.fixture(autouse=True)
    def clean_registry(self):
        yield

    @pytest.fixture(scope="class")
    def bm25_env(self):
        registry = _build_or_load_bm25_index()
        agent = _make_e2e_agent(
            "vertex_ai/claude-sonnet-4@20250514", registry, vertex_location="us-east5"
        )
        return agent, registry

    def test_routes_and_returns_relevant_results(self, bm25_env):
        """claude-sonnet-4 must route to bm25_search, execute, and return
        relevant results for a keyword query."""
        agent, registry = bm25_env
        invocations = agent.route("BM25CodeIndexer", top_k=10)

        skill_ids = [inv["skill_id"] for inv in invocations]
        assert "bm25_search" in skill_ids, (
            f"Expected bm25_search, got: {skill_ids}"
        )

        bm25_inv = next(i for i in invocations if i["skill_id"] == "bm25_search")
        executor_fn = registry.get("bm25_search").executor_fn
        results = executor_fn(**bm25_inv["inputs"])

        assert isinstance(results, list) and len(results) > 0
        names = [r.node_name for r in results]
        relevant = [n for n in names if "bm25" in n.lower()]
        assert relevant, f"No BM25-related nodes in top results: {names}"
