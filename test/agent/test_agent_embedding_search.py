"""
Agent integration tests: AgentRunner must invoke embedding_search.

Uses real FAISS over httpie/cli (shared cache with e2e tests via CODEMINER_INDEX_PATH).

Run: pytest test/agent/test_agent_embedding_search.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

import codeminer.agent.skills as _skills_pkg
from codeminer.agent.runner import AgentRunner
from codeminer.agent.skills.loader import SkillLoader
from codeminer.agent.skills.registry import SkillRegistry

SKILLS_DIR = str(Path(_skills_pkg.__file__).parent)


class TestAgentEmbeddingSearch:
    """
    Vertex (or CODEMINER_VERTEX_ROUTING_MODEL) + real FAISS; assert tool_calls use embedding_search.

    Index default: /tmp/codeminer_e2e_index — override with CODEMINER_INDEX_PATH.
    """

    @pytest.fixture(scope="class", autouse=True)
    def reset_for_class(self):
        SkillRegistry.reset()
        yield
        SkillRegistry.reset()

    @pytest.fixture(scope="class")
    def real_embedding_search(self, httpie_cli_repo):
        """Load embedding_search skill with real FAISS index over httpie/cli."""
        import os

        from codeminer.index.embedding import (
            CodeVectorStore,
            build_hierarchical_vector_store,
        )
        from codeminer.ops.retrieve import RetrieveContext

        repo_path = str(httpie_cli_repo)
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
                    "batch_size": 8,
                    "normalize_embeddings": True,
                },
            )
            vs.load(index_path)
        else:
            print(f"Building new index at {index_path}")
            Path(index_path).mkdir(parents=True, exist_ok=True)
            vs = build_hierarchical_vector_store(
                repo_path=repo_path,
                index_path=index_path,
                languages=["python"],
                embedding_model="nomic-ai/CodeRankEmbed",
                embedding_provider="huggingface",
                embedding_dimension=768,
                embedding_kwargs={
                    "model_kwargs": {"trust_remote_code": True},
                    "encode_kwargs": {
                        "batch_size": 8,
                        "normalize_embeddings": True,
                    },
                },
            )

        ctx = RetrieveContext(vector_store=vs, default_level="l2")
        loader = SkillLoader()

        skill_dir = str(Path(SKILLS_DIR) / "embedding_search")
        meta = loader.load_skill(skill_dir, contexts={"retrieve": ctx})
        assert meta is not None
        SkillRegistry().register(meta)

        return meta

    @pytest.mark.parametrize(
        "vertex_model",
        [
            "vertex_ai/gemini-2.5-flash",
            "vertex_ai/gemini-2.0-flash",
            "vertex_ai/claude-sonnet-4@20250514",
        ],
    )
    def test_vertex_agent_selects_embedding_search(
        self, real_embedding_search, vertex_model
    ):
        """Vertex AI agent must select embedding_search for a semantic query."""
        import os

        model_name = os.environ.get("CODEMINER_VERTEX_ROUTING_MODEL", vertex_model)

        try:
            runner = AgentRunner(
                model=model_name,
                registry=SkillRegistry(),
                max_turns=3,
            )
            result = runner.run(
                "function that sends an HTTP request with authentication",
            )
        except Exception as exc:
            pytest.skip(f"Vertex agent unavailable for {model_name}: {exc}")

        assert result.total_turns >= 1
        assert len(result.tool_calls) >= 1

        skill_ids = [tc.skill_id for tc in result.tool_calls]
        assert (
            "embedding_search" in skill_ids
        ), f"Expected embedding_search for {model_name}, got: {skill_ids}"
