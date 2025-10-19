#!/usr/bin/env python3
"""
Test script for simple repository code chunking functionality.
"""

import argparse
from pathlib import Path

from codeminer.code_chunker import CodeChunker
from codeminer.env import load_filter_locbench_dataset, process_locbench_instance


def test_simple_repo_chunker():
    """Test simple repository code chunking."""
    print("\n=== Testing Simple Repository Chunker ===")

    try:
        # Create chunker with default config
        chunker = CodeChunker(language="python")

        # Get the project root (parent of this test file)
        project_root = Path(__file__).parent.parent
        codeminer_path = project_root / "codeminer"

        if codeminer_path.exists():
            # Test repository stats
            print("Getting repository stats...")
            stats = chunker.get_repository_stats(str(codeminer_path))

            print(f"Repository: {stats['repo_path']}")
            print(f"Total files: {stats['total_files']}")
            print(f"Total size: {stats['total_size_mb']} MB")

            # Test repository chunking
            print("\nChunking repository...")
            chunks = chunker.chunk_repository(str(codeminer_path))

            print(f"Generated {len(chunks)} chunks from repository")

            # Test repository summary
            chunker.print_repository_summary(chunks)

            # Show some example chunks
            print(f"\n--- Example Repository Chunks ---")
            for i, chunk in enumerate(chunks[:2]):
                print(f"\nChunk {i+1} from repository:")
                print(f"  File: {Path(chunk.file).name}")
                print(f"  Type: {chunk.chunk_type}")
                print(f"  Name: {chunk.name}")
                print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
                print(f"  Content preview: {chunk.content[:80]}...")

        print("✅ Simple repository chunker test passed!")

    except Exception as e:
        print(f"❌ Simple repository chunker test failed: {e}")
        import traceback

        traceback.print_exc()


args_dict = {
    "model": "gpt-4o",
    "dataset": "czlll/Loc-Bench_V1",
    "split": "test",
    "filter_instance": "^(sympy__sympy-27223)$",
}


def test_sympy_repo_chunker():
    args = argparse.Namespace(**args_dict)
    dataset = load_filter_locbench_dataset(args=args)
    # process the first instance
    instance = dataset[0]
    repo_path = process_locbench_instance(instance)
    print(f"\n=== Testing SymPy Repository Chunker ===")

    chunker = CodeChunker(language="python")
    if Path(repo_path).exists():
        chunks = chunker.chunk_repository(str(repo_path))
        print(f"Generated {len(chunks)} chunks from SymPy repository")

        # chunker.print_repository_summary(chunks)

    # get chunk name in "sympy/utilities/lambdify.py"
    sympy_chunks = [
        chunk for chunk in chunks if "sympy/utilities/lambdify.py" in chunk.file
    ]
    print(f"Chunks in sympy/utilities/lambdify.py: {len(sympy_chunks)}")
    for chunk in sympy_chunks:
        print(f"  Node id: {chunk.node_id}, Lines: {chunk.start_line}-{chunk.end_line}")
    print("✅ SymPy repository chunker test passed!")
