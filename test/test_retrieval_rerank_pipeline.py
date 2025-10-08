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
    
    assert pipeline is not None
    print(pipeline.vector_store.get_stats())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
