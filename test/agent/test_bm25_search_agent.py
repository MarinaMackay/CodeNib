"""
Agent integration tests for bm25_search skill.

TestBM25SearchAgentE2E: real BM25 index over httpie/cli + Vertex AI agent routing.
Marked @pytest.mark.slow, run with: pytest -v -m slow
"""

from __future__ import annotations

from pathlib import Path

import pytest

import codeminer.agent.skills as _skills_pkg
from codeminer.agent.runner import AgentRunner
from codeminer.agent.skills.loader import SkillLoader
from codeminer.agent.skills.registry import SkillRegistry

SKILLS_DIR = str(Path(_skills_pkg.__file__).parent)


@pytest.mark.slow
class TestBM25SearchAgentE2E:
    """
    Full round-trip: real BM25 index + Vertex AI agent routing.

    BM25 index is cached at /tmp/codeminer_bm25_e2e_index (reused across runs).
    Set CODEMINER_BM25_INDEX_PATH to override.
    """

    @pytest.fixture(scope="class", autouse=True)
    def reset_for_class(self):
        SkillRegistry.reset()
        yield
        SkillRegistry.reset()

    @pytest.fixture(scope="class")
    def real_bm25_search(self, httpie_cli_repo):
        """Load bm25_search skill with real BM25 index over httpie/cli."""
        import os

        from codeminer.code_chunker import CodeChunker, RepoChunkingConfig
        from codeminer.index.sparse_idx.bm25_index import BM25CodeIndexer
        from codeminer.ops.retrieve import RetrieveContext

        repo_path = str(httpie_cli_repo)
        index_path = os.environ.get(
            "CODEMINER_BM25_INDEX_PATH", "/tmp/codeminer_bm25_e2e_index"
        )
        documents_file = Path(index_path) / "documents.json"

        if documents_file.exists():
            print(f"Loading existing BM25 index from {index_path}")
            indexer = BM25CodeIndexer(max_k=128)
            indexer.load_index(index_path)
        else:
            print(f"Building new BM25 index at {index_path}")
            chunker = CodeChunker(
                language="python",
                repo_config=RepoChunkingConfig(languages=["python"]),
                max_lines_per_chunk=300,
            )
            chunks = chunker.chunk_repository(repo_path=repo_path)
            assert chunks, "No code chunks generated"
            indexer = BM25CodeIndexer(chunks=chunks, max_k=128)
            indexer.project_root = repo_path
            Path(index_path).mkdir(parents=True, exist_ok=True)
            indexer.save_index(index_path)

        ctx = RetrieveContext(bm25=indexer)
        loader = SkillLoader()

        skill_dir = str(Path(SKILLS_DIR) / "bm25_search")
        meta = loader.load_skill(skill_dir, contexts={"retrieve": ctx})
        assert meta is not None
        SkillRegistry().register(meta)

        return meta

    def test_bm25_search_returns_results(self, real_bm25_search):
        """bm25_search must return results for a keyword query."""
        results = real_bm25_search.executor_fn(
            query="HTTPie",
            top_k=5,
        )
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.parametrize(
        "vertex_model",
        [
            "vertex_ai/gemini-2.5-flash",
            "vertex_ai/gemini-2.0-flash",
            "vertex_ai/claude-sonnet-4@20250514",
        ],
    )
    def test_vertex_agent_selects_bm25_search(self, real_bm25_search, vertex_model):
        """Vertex AI agent must select bm25_search for a keyword query."""
        import os

        model_name = os.environ.get("CODEMINER_VERTEX_ROUTING_MODEL", vertex_model)

        try:
            runner = AgentRunner(
                model=model_name,
                registry=SkillRegistry(),
                max_turns=3,
            )
            result = runner.run("HTTPie class definition")
        except Exception as exc:
            pytest.skip(f"Vertex agent unavailable for {model_name}: {exc}")

        assert result.total_turns >= 1
        assert len(result.tool_calls) >= 1

        skill_ids = [tc.skill_id for tc in result.tool_calls]
        assert (
            "bm25_search" in skill_ids
        ), f"Expected bm25_search for {model_name}, got: {skill_ids}"
