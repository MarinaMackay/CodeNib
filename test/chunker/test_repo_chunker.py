#!/usr/bin/env python3
"""
Test script for simple repository code chunking functionality.
"""

import argparse
from pathlib import Path

from codeminer.code_chunker import CodeChunker
from codeminer.env import load_filter_locbench_dataset, process_locbench_instance

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
