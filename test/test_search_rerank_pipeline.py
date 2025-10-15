import argparse
import os
import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples import SearchRerankPipeline
from codeminer.env.process_locbench_data import (
    load_filter_locbench_dataset,
    process_locbench_instance,
)
from codeminer.env.process_swebench_data import (
    load_filter_swebench_dataset,
    process_swebench_instance,
)

# start the vLLM server in a separate terminal before running this script
# python scripts/start_vllm_server.py --model Qwen/Qwen2.5-Coder-7B

PIPELINE_DEFAULTS = {
    "embedding_model": "nomic-ai/CodeRankEmbed",
    "embedding_provider": "huggingface",
    "embedding_dimension": 768,
    "embedding_model_kwargs": {
        "trust_remote_code": True,
        "encode_kwargs": {
            "batch_size": 8,  # Small batch size to avoid OOM
        },
    },
    "rerank_model": "Qwen/Qwen2.5-Coder-7B",
    "languages": ["python"],
    "max_lines_per_chunk": 100,
}


def _run_pipeline_test(dataset_name, loader, processor, args_kwargs, pipeline_overrides=None):
    """Helper to run the search + rerank pipeline test for a dataset."""
    args = argparse.Namespace(**args_kwargs)

    try:
        dataset = loader(args=args)
    except Exception as exc:
        pytest.skip(f"Skipping {dataset_name} dataset: {exc}")

    if len(dataset) == 0:
        pytest.skip(f"No instances returned for {dataset_name}")

    for _, instance in enumerate(dataset):
        print("=" * 80)
        print(
            f"[{dataset_name}] Loaded instance: {instance['instance_id']} from repo {instance['repo']}"
        )
        print(f"Base commit: {instance['base_commit']}")
        print(f"Problem statement: {instance['problem_statement'][:200]}...")
        print("=" * 80)

        repo_path = processor(instance)
        print(f"\nRepository path: {repo_path}")

        print("\nInitializing SearchRerankPipeline...")
        pipeline_kwargs = PIPELINE_DEFAULTS.copy()
        if pipeline_overrides:
            pipeline_kwargs.update(pipeline_overrides)

        pipeline = SearchRerankPipeline(
            repo_path=repo_path,
            **pipeline_kwargs,
        )

        assert pipeline is not None
        stats = pipeline.vector_store.get_stats()
        print(f"Vector store stats: {stats}")

        query = instance["problem_statement"]
        print(f"\nQuerying with problem statement (top 5 results)...")
        results = pipeline.query(query=query, top_k=5)
        assert isinstance(results, list)
        assert len(results) > 0

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
        print(
            f"Test completed successfully for [{dataset_name}] instance: {instance['instance_id']}"
        )
        print("=" * 80)


def test_swebench_search_rerank_pipeline():
    """
    Test search + rerank pipeline on a SWE-bench Lite instance.
    """
    args_kwargs = {
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "split": "test",
        "filter_instance": "^(astropy__astropy-12907)$",
    }
    _run_pipeline_test(
        dataset_name="swebench",
        loader=load_filter_swebench_dataset,
        processor=process_swebench_instance,
        args_kwargs=args_kwargs,
    )


def test_locbench_search_rerank_pipeline():
    """
    Test search + rerank pipeline on a LocBench instance.
    """
    args_kwargs = {
        "dataset": "czlll/Loc-Bench_V0.2",
        "split": "test",
        "filter_instance": "^(joselc__life-sim-first-try-2)$",
    }
    _run_pipeline_test(
        dataset_name="locbench",
        loader=load_filter_locbench_dataset,
        processor=process_locbench_instance,
        args_kwargs=args_kwargs,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
