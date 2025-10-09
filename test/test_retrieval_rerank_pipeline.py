import os
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples import SearchRerankPipeline


def test_search_rerank_pipeline():
    
    repo_path = Path(__file__).parent / "simple_repo"
    pipeline = SearchRerankPipeline(
        repo_path=repo_path,
        embedding_model="microsoft/unixcoder-base",
        embedding_provider="huggingface",
        embedding_dimension=768,
        rerank_model="Salesforce/SweRankLLM-Small",
        rerank_provider="vllm",
    )
    
    # Check pipeline vector store stats
    assert pipeline is not None
    print(pipeline.vector_store.get_stats())

    # Check pipeline search with content
    query = "Calculator multiply two numbers"
    results = pipeline.vector_store.search_with_content(query=query, top_k=5)
    assert isinstance(results, list)
    print(results[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
