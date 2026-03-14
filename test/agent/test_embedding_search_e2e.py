"""
End-to-end smoke test for the embedding_search skill.

This test builds (or reloads) a real FAISS index over the CodeMiner repository
itself, then exercises the full skill loading and execution path.

Marked with @pytest.mark.slow so it is excluded from the default test run.
To run explicitly:

    pytest test/agent/test_embedding_search_e2e.py -v -m slow

The index is written to /tmp/codeminer_e2e_index/ and is reused across runs.
Delete that directory to force a rebuild.

Environment variable CODEMINER_INDEX_PATH can override the cache location:

    CODEMINER_INDEX_PATH=/my/path pytest ... -m slow
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default lightweight model that runs on CPU without an API key.
# nomic-ai/CodeRankEmbed is used in examples/skill_agent.py.
DEFAULT_EMBEDDING_MODEL = "nomic-ai/CodeRankEmbed"
DEFAULT_EMBEDDING_DIM = 768
DEFAULT_EMBEDDING_PROVIDER = "huggingface"

SKILL_DIR = str(_PROJECT_ROOT / "codeminer" / "agent" / "skills" / "embedding_search")


@pytest.fixture(scope="session")
def index_path() -> str:
    """Return the directory used to cache the FAISS index."""
    env_override = os.environ.get("CODEMINER_INDEX_PATH")
    if env_override:
        return env_override
    return "/tmp/codeminer_e2e_index"


@pytest.fixture(scope="session")
def vector_store(index_path: str):
    """
    Build a CodeVectorStore for the CodeMiner repo, or load it from cache.

    Building takes 2-5 minutes the first time (embedding all Python files).
    Subsequent runs reuse the cached FAISS index and finish in seconds.
    """
    from codeminer.index.embedding import (
        CodeVectorStore,
        build_hierarchical_vector_store,
    )

    store_root = Path(index_path)
    l0_dir = store_root / "l0"
    l2_dir = store_root / "l2"

    if l0_dir.exists() and l2_dir.exists():
        # Reload cached index.
        print(f"\n[e2e] Loading cached index from {index_path}")
        vs = CodeVectorStore(
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            embedding_provider=DEFAULT_EMBEDDING_PROVIDER,
            dimension=DEFAULT_EMBEDDING_DIM,
            store_path=index_path,
            model_kwargs={
                "trust_remote_code": True,
                "device": "cuda",
                "model_kwargs": {"torch_dtype": "float16"},  # Use half precision
            },
            encode_kwargs={
                "batch_size": 4,  # Further reduce batch size
                "normalize_embeddings": True,
            },
        )
        vs.load(index_path)
    else:
        # Build from scratch.
        print(f"\n[e2e] Building index for {_PROJECT_ROOT} into {index_path}")
        store_root.mkdir(parents=True, exist_ok=True)
        vs = build_hierarchical_vector_store(
            repo_path=str(_PROJECT_ROOT),
            index_path=index_path,
            languages=["python"],
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            embedding_provider=DEFAULT_EMBEDDING_PROVIDER,
            embedding_dimension=DEFAULT_EMBEDDING_DIM,
            embedding_kwargs={
                "model_kwargs": {
                    "trust_remote_code": True,
                    "device": "cuda",
                    "model_kwargs": {"torch_dtype": "float16"},  # Use half precision
                },
                "encode_kwargs": {
                    "batch_size": 4,  # Further reduce batch size
                    "normalize_embeddings": True,
                },
            },
        )

    return vs


@pytest.fixture(scope="session")
def retrieve_context(vector_store):
    """Construct a real RetrieveContext backed by the session-scoped vector store."""
    from codeminer.ops.retrieve import RetrieveContext

    return RetrieveContext(vector_store=vector_store)


@pytest.fixture(scope="session")
def executor_fn(retrieve_context):
    """
    Load the embedding_search skill and return the bound executor callable.

    The SkillLoader reads config.yaml, skill.md, and executor.py from the
    real package directory, then calls create_executor(context) to produce
    the executor closure.
    """
    from codeminer.agent.skills.loader import SkillLoader
    from codeminer.agent.skills.registry import SkillRegistry

    SkillRegistry.reset()
    loader = SkillLoader()
    meta = loader.load_skill(SKILL_DIR, contexts={"retrieve": retrieve_context})
    assert meta is not None, f"SkillLoader returned None for {SKILL_DIR}"
    assert meta.executor_fn is not None, "executor_fn is None after loading"
    return meta.executor_fn


@pytest.mark.slow
class TestEmbeddingSearchE2E:
    """Full-stack tests using a real FAISS index."""

    def setup_method(self):
        """Check GPU memory before each test."""
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\n[GPU] Before test: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def teardown_method(self):
        """Clean up GPU memory after each test."""
        import gc
        import torch

        # Force garbage collection
        gc.collect()

        # Clear PyTorch GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @classmethod
    def teardown_class(cls):
        """Clean up GPU memory after all tests complete."""
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_basic_query_returns_list(self, executor_fn):
        """A plain query must return a non-empty list."""
        results = executor_fn("skill registry lookup", top_k=5)
        assert isinstance(results, list)
        assert len(results) > 0, "Expected at least one result for a broad query"

    def test_results_respect_top_k(self, executor_fn):
        """Result count must not exceed top_k."""
        top_k = 3
        results = executor_fn("search query", top_k=top_k)
        assert len(results) <= top_k

    def test_results_have_score(self, executor_fn):
        """Every returned node must carry a numeric similarity score."""
        results = executor_fn("code chunker", top_k=5)
        for node in results:
            assert hasattr(node, "score"), f"Node missing 'score': {node}"
            assert isinstance(node.score, float)

    def test_results_have_file(self, executor_fn):
        """Every result must reference a source file."""
        results = executor_fn("class definition", top_k=5)
        for node in results:
            assert hasattr(node, "file"), f"Node missing 'file': {node}"
            assert node.file, "file field is empty"

    def test_return_content_true_populates_content(self, executor_fn):
        """With return_content=True, each node must have non-empty content."""
        results = executor_fn("vector store search", top_k=3, return_content=True)
        assert len(results) > 0
        for node in results:
            assert hasattr(node, "content"), f"Node missing 'content': {node}"
            assert node.content, "content field is empty"

    def test_return_content_false_omits_content(self, executor_fn):
        """With return_content=False, content field should be absent or empty."""
        results = executor_fn("vector store search", top_k=3, return_content=False)
        assert len(results) > 0
        for node in results:
            # NodeInfo.content is Optional; it may be None or absent.
            content = getattr(node, "content", None)
            assert not content, (
                f"Expected no content with return_content=False, got: {content[:40]}"
            )

    def test_l0_level_returns_results(self, executor_fn):
        """level='l0' (file skeletons) must also return results."""
        results = executor_fn("embedding index", top_k=5, level="l0")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_l2_level_returns_results(self, executor_fn):
        """level='l2' (function/method chunks) must return results."""
        results = executor_fn("embedding index", top_k=5, level="l2")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_score_threshold_filters_low_scores(self, executor_fn):
        """A very high threshold should return fewer results than no threshold."""
        all_results = executor_fn("code", top_k=20)
        filtered = executor_fn("code", top_k=20, score_threshold=0.999)
        assert len(filtered) <= len(all_results)

    def test_conceptual_query_finds_relevant_file(self, executor_fn):
        """A semantic query should surface files related to the concept."""
        results = executor_fn(
            "function that registers a skill into a catalogue", top_k=10
        )
        assert len(results) > 0
        # At least one result should mention registry or skill in its file path.
        files = [node.file for node in results]
        relevant = [f for f in files if "skill" in f or "registry" in f]
        assert relevant, (
            f"No skill/registry file in top-10 results. Got: {files}"
        )

    def test_results_have_start_and_end_line(self, executor_fn):
        """Every result must have line number metadata."""
        results = executor_fn("parse config", top_k=5, return_content=False)
        for node in results:
            assert hasattr(node, "start_line")
            assert hasattr(node, "end_line")
            assert isinstance(node.start_line, int)
            assert isinstance(node.end_line, int)
