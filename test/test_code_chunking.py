#!/usr/bin/env python3
"""
Test script for the code chunking functionality.
"""

import sys
from pathlib import Path

from codeminer.code_chunker import CodeChunker


def test_python_chunking():
    """Test chunking on a Python file."""
    print("=== Testing Python Code Chunking ===")

    # Path to the test file
    test_file = Path(__file__).parent / "sample_python_file.py"

    # Create chunker for Python
    chunker = CodeChunker(language="python")

    # Chunk the file
    chunks = chunker.chunk_file(str(test_file))

    if not chunks:
        print("ERROR: No chunks generated!")
        return False

    # Print detailed results
    print(f"\nGenerated {len(chunks)} chunks:")

    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Type: {chunk.chunk_type}")
        print(f"Name: {chunk.name}")
        print(f"Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"Content preview (first 100 chars):")
        print(f"'{chunk.content[:100]}{'...' if len(chunk.content) > 100 else ''}'")

    # Print summary
    chunker.print_chunk_summary(chunks)

    # Save to JSON
    output_file = Path(__file__).parent / "test_chunks_output.json"
    chunker.save_chunks_to_json(chunks, str(output_file))

    # Verify expected chunks - now includes methods within classes
    # Note: header and epilogue are excluded by default (include_header_epilogue=False)
    # Note: class chunks are excluded by default (include_class_level=False, aligned with SweRank)
    expected_types = [
        "function",  # hello_world
        "method",  # Calculator.__init__
        "method",  # Calculator.add
        "method",  # Calculator.subtract
        "function",  # fibonacci
        "method",  # DataProcessor.__init__
        "method",  # DataProcessor.process
        "function",  # async_function
    ]
    actual_types = [chunk.chunk_type for chunk in chunks]

    print(f"\nExpected chunk types: {expected_types}")
    print(f"Actual chunk types: {actual_types}")

    # Also show chunk names for better debugging
    # Note: method names now include class prefix (e.g., Calculator.__init__)
    expected_names = [
        "hello_world",
        "Calculator.__init__",
        "Calculator.add",
        "Calculator.subtract",
        "fibonacci",
        "DataProcessor.__init__",
        "DataProcessor.process",
        "async_function",
    ]
    actual_names = [chunk.name for chunk in chunks]

    print(f"\nExpected chunk names: {expected_names}")
    print(f"Actual chunk names: {actual_names}")

    if actual_types == expected_types and actual_names == expected_names:
        print("=== Chunk types and names match expected pattern! ===")
        return True
    else:
        print("=== Chunk types or names don't match expected pattern! ===")
        if actual_types != expected_types:
            print("Type mismatch!")
        if actual_names != expected_names:
            print("Name mismatch!")
        return False


def test_python_chunking_with_classes():
    """Test chunking on a Python file with include_class_level=True."""
    print("\n=== Testing Python Code Chunking with include_class_level=True ===")

    # Path to the test file
    test_file = Path(__file__).parent / "sample_python_file.py"

    # Create chunker for Python with include_class_level=True
    chunker = CodeChunker(language="python", include_class_level=True)

    # Chunk the file
    chunks = chunker.chunk_file(str(test_file))

    if not chunks:
        print("ERROR: No chunks generated!")
        return False

    # Print detailed results
    print(f"\nGenerated {len(chunks)} chunks:")

    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Type: {chunk.chunk_type}")
        print(f"Name: {chunk.name}")

    # Verify that class chunks are now included
    expected_types = [
        "function",  # hello_world
        "class",  # Calculator class
        "method",  # Calculator.__init__
        "method",  # Calculator.add
        "method",  # Calculator.subtract
        "function",  # fibonacci
        "class",  # DataProcessor class
        "method",  # DataProcessor.__init__
        "method",  # DataProcessor.process
        "function",  # async_function
    ]
    actual_types = [chunk.chunk_type for chunk in chunks]

    print(f"\nExpected chunk types: {expected_types}")
    print(f"Actual chunk types: {actual_types}")

    if actual_types == expected_types:
        print("=== Class chunks correctly included when include_class_level=True! ===")
        return True
    else:
        print("=== Unexpected chunk types! ===")
        return False


def main():
    """Run all tests."""
    print("Starting code chunking tests...\n")

    success = True

    # Test Python chunking (default: no class chunks)
    if not test_python_chunking():
        success = False

    # Test Python chunking with class chunks enabled
    if not test_python_chunking_with_classes():
        success = False

    if success:
        print("\n=== All tests passed! ===")
    else:
        print("\n=== Some tests failed! ===")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
