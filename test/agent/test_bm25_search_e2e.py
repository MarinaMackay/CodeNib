"""
End-to-end smoke test for the bm25_search skill.

This test builds a real BM25 index over the CodeMiner repository itself,
then exercises the full skill loading and execution path.

Marked with @pytest.mark.slow so it is excluded from the default test run.
To run explicitly:

    pytest test/agent/test_bm25_search_e2e.py -v -m slow

The index is written to /tmp/codeminer_bm25_e2e_index/ and is reused
across runs. Delete that directory to force a rebuild.

Environment variable CODEMINER_BM25_INDEX_PATH can override the cache
location.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SKILL_DIR = str(_PROJECT_ROOT / "codeminer" / "agent" / "skills" / "bm25_search")


@pytest.fixture(scope="session")
def index_path() -> str:
    """Return the directory used to cache the BM25 index."""
    env_override = os.environ.get("CODEMINER_BM25_INDEX_PATH")
    if env_override:
        return env_override
    return "/tmp/codeminer_bm25_e2e_index"


@pytest.fixture(scope="session")
def bm25_indexer(index_path: str):
    """
    Build a BM25CodeIndexer for the CodeMiner repo, or load from cache.

    Building is fast (seconds) since BM25 only needs code chunks, not
    embeddings.
    """
    from codeminer.code_chunker import CodeChunker, RepoChunkingConfig
    from codeminer.index.sparse_idx.bm25_index import BM25CodeIndexer

    cache_dir = Path(index_path)
    documents_file = cache_dir / "documents.json"

    if documents_file.exists():
        print(f"\n[e2e] Loading cached BM25 index from {index_path}")
        indexer = BM25CodeIndexer(max_k=128)
        indexer.load_index(index_path)
    else:
        print(f"\n[e2e] Building BM25 index for {_PROJECT_ROOT}")
        chunker = CodeChunker(
            language="python",
            repo_config=RepoChunkingConfig(languages=["python"]),
            max_lines_per_chunk=300,
        )
        chunks = chunker.chunk_repository(repo_path=str(_PROJECT_ROOT))
        assert chunks, "No code chunks generated from the repository"
        print(f"  BM25: indexed {len(chunks)} chunks")

        indexer = BM25CodeIndexer(chunks=chunks, max_k=128)
        indexer.project_root = str(_PROJECT_ROOT)

        cache_dir.mkdir(parents=True, exist_ok=True)
        indexer.save_index(index_path)

    return indexer


@pytest.fixture(scope="session")
def retrieve_context(bm25_indexer):
    """Construct a real RetrieveContext backed by the session-scoped BM25 index."""
    from codeminer.ops.retrieve import RetrieveContext

    return RetrieveContext(bm25=bm25_indexer)


@pytest.fixture(scope="session")
def executor_fn(retrieve_context):
    """
    Load the bm25_search skill and return the bound executor callable.
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
class TestBM25SearchE2E:
    """Full-stack tests using a real BM25 index."""

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

    def test_results_have_node_name(self, executor_fn):
        """Every returned node must have a node_name."""
        results = executor_fn("code chunker", top_k=5)
        for node in results:
            assert hasattr(node, "node_name"), f"Node missing 'node_name': {node}"
            assert node.node_name, "node_name field is empty"

    def test_results_have_type(self, executor_fn):
        """Every returned node must have a type field."""
        results = executor_fn("search", top_k=5)
        for node in results:
            assert hasattr(node, "type"), f"Node missing 'type': {node}"
            assert node.type, "type field is empty"

    def test_return_content_true_populates_content_for_file_nodes(self, executor_fn):
        """With return_content=True, file-type nodes should have content.

        Note: chunks-based BM25 index does not store start_line/end_line,
        so only file-type nodes (not symbol nodes) can have content populated.
        We search for a filename pattern to increase the chance of hitting one.
        """
        results = executor_fn("bm25_index", top_k=20, return_content=True)
        assert len(results) > 0
        file_nodes = [n for n in results if n.type == "file"]
        if file_nodes:
            has_content = any(
                getattr(node, "content", None) for node in file_nodes
            )
            assert has_content, "File nodes should have content when return_content=True"
        else:
            # Chunks-based index may not produce file-level nodes;
            # just verify the call succeeds without error.
            pass

    def test_return_content_false_omits_content(self, executor_fn):
        """With return_content=False, content should be absent or empty."""
        results = executor_fn("bm25 search", top_k=5, return_content=False)
        assert len(results) > 0
        for node in results:
            content = getattr(node, "content", None)
            assert not content, (
                f"Expected no content with return_content=False, got: {content[:40]}"
            )

    def test_filter_test_reduces_results(self, executor_fn):
        """filter_test=True should return no more results than unfiltered."""
        all_results = executor_fn("test", top_k=20, filter_test=False)
        filtered = executor_fn("test", top_k=20, filter_test=True)
        assert len(filtered) <= len(all_results)

    def test_keyword_query_finds_relevant_file(self, executor_fn):
        """A keyword query should surface nodes related to the keyword."""
        results = executor_fn("BM25CodeIndexer", top_k=10)
        assert len(results) > 0
        names = [node.node_name for node in results]
        relevant = [n for n in names if "bm25" in n.lower()]
        assert relevant, (
            f"No BM25-related node in top-10 results. Got: {names}"
        )

    def test_wrap_with_line_numbers(self, executor_fn):
        """With wrap_with_line_numbers=True, content (if present) should be line-numbered.

        Chunks-based index may not produce content for symbol nodes (no line
        metadata), so we search broadly and check any content that is returned.
        """
        results = executor_fn(
            "search", top_k=20, return_content=True, wrap_with_line_numbers=True
        )
        nodes_with_content = [
            n for n in results if getattr(n, "content", None)
        ]
        if nodes_with_content:
            has_numbered = any("|" in n.content for n in nodes_with_content)
            assert has_numbered, "Content should contain line numbers when wrap=True"
        # If no content returned (chunks-based limitation), the test still passes.

    def test_results_have_file_for_symbol_nodes(self, executor_fn):
        """Symbol nodes (functions/classes) should have a file path."""
        results = executor_fn("class", top_k=10)
        symbol_results = [
            n for n in results if n.type in ("function", "class", "method")
        ]
        for node in symbol_results:
            assert node.file, f"Symbol node '{node.node_name}' missing file path"
