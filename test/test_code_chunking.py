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

    # Verify expected chunks
    expected_types = [
        "header",
        "function",
        "class",
        "function",
        "class",
        "function",
        "epilogue",
    ]
    actual_types = [chunk.chunk_type for chunk in chunks]

    print(f"\nExpected chunk types: {expected_types}")
    print(f"Actual chunk types: {actual_types}")

    if actual_types == expected_types:
        print("=== Chunk types match expected pattern! ===")
        return True
    else:
        print("=== Chunk types don't match expected pattern! ===")
        return False


def main():
    """Run all tests."""
    print("Starting code chunking tests...\n")

    success = True

    # Test Python chunking
    if not test_python_chunking():
        success = False

    if success:
        print("\n=== All tests passed! ===")
    else:
        print("\n=== Some tests failed! ===")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
