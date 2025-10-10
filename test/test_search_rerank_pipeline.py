import argparse
import os
import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples import SearchRerankPipeline
from codeminer.env.process_data import (
    load_filter_swebench_dataset,
    process_swebench_instance,
)

args_dict = {
    "dataset": "princeton-nlp/SWE-bench_Lite",
    "split": "test",
    "filter_instance": "^(astropy__astropy-12907)$",
}


def test_search_rerank_pipeline():
    """
    Test search + rerank pipeline on SWE-bench instance.
    
    Workflow:
    1. Load SWE-bench instance (astropy__astropy-12907)
    2. Initialize SearchRerankPipeline with embedding and rerank models
    3. Query with problem statement and get top 5 nodes
    4. Display detailed results
    """
    
    # Load instance from SWE-bench dataset
    args = argparse.Namespace(**args_dict)
    dataset = load_filter_swebench_dataset(args=args)
    
    for _, instance in enumerate(dataset):
        print("=" * 80)
        print(f"Loaded instance: {instance['instance_id']} from repo {instance['repo']}")
        print(f"Base commit: {instance['base_commit']}")
        print(f"Problem statement: {instance['problem_statement'][:200]}...")
        print("=" * 80)
        
        # Process instance and get repo path
        repo_path = process_swebench_instance(instance)
        print(f"\nRepository path: {repo_path}")
        
        # Initialize SearchRerankPipeline with CodeRankEmbed and CodeRankLLM
        print("\nInitializing SearchRerankPipeline...")
        pipeline = SearchRerankPipeline(
            repo_path=repo_path,
            embedding_model="nomic-ai/CodeRankEmbed",
            embedding_provider="huggingface",
            embedding_dimension=768,
            embedding_model_kwargs={
                "trust_remote_code": True,
                "encode_kwargs": {
                    "batch_size": 8,  # Small batch size to avoid OOM
                }
            },
            rerank_model="nomic-ai/CodeRankLLM",
            languages=['python'],
            max_lines_per_chunk=100,
        )
        
        # Check pipeline initialization
        assert pipeline is not None
        stats = pipeline.vector_store.get_stats()
        print(f"Vector store stats: {stats}")
        
        # Query with problem statement and get top 5 nodes
        query = instance["problem_statement"]
        print(f"\nQuerying with problem statement (top 5 results)...")
        results = pipeline.query(query=query, top_k=5)
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Detailed results display
        print("\n" + "=" * 80)
        print(f"Top {len(results)} ranked nodes:")
        print("=" * 80)
        
        for i, node in enumerate(results):
            print(f"\n{'─' * 80}")
            print(f"Rank {i+1} (Score: {node.score:.4f})")
            print(f"{'─' * 80}")
            print(f"  Node Name: {node.node_name}")
            print(f"  Node Type: {node.type}")
            print(f"  File: {node.file}")
            print(f"  Lines: {node.start_line}-{node.end_line}")
            print()
        
        print("=" * 80)
        print(f"Test completed successfully for instance: {instance['instance_id']}")
        print("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
