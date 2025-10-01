#!/usr/bin/env python3
"""
Test script for the merged CodeChunker functionality.
"""

import argparse
import sys
from pathlib import Path

from codeminer.code_chunker import CodeChunker, RepoChunkingConfig
from codeminer.env.process_data import load_filter_swebench_dataset

# Add the parent directory to the path to import codeminer modules
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_repository_functionality():
    """Test the new repository-level functionality."""
    print("\n=== Testing Repository Functionality ===")

    try:
        # Test with default config
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

        print("✅ Repository functionality test passed!")

    except Exception as e:
        print(f"❌ Repository functionality test failed: {e}")
        import traceback

        traceback.print_exc()


def test_custom_config():
    """Test the repository functionality with custom configuration."""
    print("\n=== Testing Custom Configuration ===")

    try:
        # Create custom config
        config = RepoChunkingConfig(
            max_file_size_mb=5.0,
            ignore_dirs={"__pycache__", ".git", "test"},  # Ignore test directory
        )

        chunker = CodeChunker(language="python", repo_config=config)

        project_root = Path(__file__).parent.parent
        codeminer_path = project_root / "codeminer"

        if codeminer_path.exists():
            print("Testing with custom config...")
            chunks = chunker.chunk_repository(str(codeminer_path))

            print(f"Generated {len(chunks)} chunks with custom config")

            # Check that test files were ignored (if any exist in codeminer/)
            test_files = [chunk for chunk in chunks if "test" in chunk.file.lower()]
            print(
                f"Test files found (should be 0 with ignore config): {len(test_files)}"
            )

        print("✅ Custom configuration test passed!")

    except Exception as e:
        print(f"❌ Custom configuration test failed: {e}")


def test_swebench_functionality():
    """Test the SWE-bench repository processing functionality."""
    print("\n=== Testing SWE-bench Functionality ===")

    try:
        # Create chunker
        chunker = CodeChunker(language="python")

        # Load a specific SWE-bench instance (using the same as test_codegraph.py)
        args_dict = {
            "model": "gpt-4o",
            "dataset": "princeton-nlp/SWE-bench_Lite",
            "split": "test",
            "filter_instance": "^(astropy__astropy-12907)$",
        }

        print("Loading SWE-bench instance...")
        args = argparse.Namespace(**args_dict)
        dataset = load_filter_swebench_dataset(args=args)

        # Process the first (and only) instance
        for instance in dataset:
            print(f"Processing SWE-bench instance: {instance['instance_id']}")

            # Use the new SWE-bench chunking functionality
            result = chunker.chunk_swebench_instance(instance)

            print(f"\nSWE-bench Chunking Results:")
            print(f"Instance ID: {result['instance_id']}")
            print(f"Repository: {result['repo']}")
            print(f"Languages detected: {result['languages']}")
            print(f"Total chunks: {result['total_chunks']}")
            print(f"Repository path: {result['repo_path']}")

            # Print chunk summary
            summary = result["chunk_summary"]
            print(f"\nChunk Summary:")
            print(f"  Files processed: {summary['files_processed']}")
            print(f"  Chunk types: {summary['chunk_types']}")

            # Show some example chunks
            chunks = result["chunks"]
            print(f"\n--- Example SWE-bench Chunks ---")
            for i, chunk in enumerate(chunks[:3]):
                print(f"\nChunk {i+1}:")
                print(
                    f"  File: {Path(chunk.file).relative_to(Path(result['repo_path']))}"
                )
                print(f"  Type: {chunk.chunk_type}")
                print(f"  Name: {chunk.name}")
                print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
                print(
                    f"  Content preview: {chunk.content[:100].replace(chr(10), ' ')}..."
                )

            # Save the result
            output_path = (
                Path(__file__).parent / f"swebench_chunks_{result['instance_id']}.json"
            )
            chunker.save_swebench_chunks(result, str(output_path))

            # Also save all chunks to JSON using the regular method
            all_chunks_path = (
                Path(__file__).parent
                / f"swebench_all_chunks_{result['instance_id']}.json"
            )
            chunker.save_chunks_to_json(chunks, str(all_chunks_path))

            print(f"\nResults saved to:")
            print(f"  Summary: {output_path}")
            print(f"  All chunks: {all_chunks_path}")

            break  # Only process the first instance

        print("✅ SWE-bench functionality test passed!")

    except Exception as e:
        print(f"❌ SWE-bench functionality test failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main test function."""
    print("Testing repo chunker functionality...\n")

    # Run all tests
    test_repository_functionality()
    test_custom_config()
    test_swebench_functionality()

    print("\n🎉 All repo chunker tests completed!")


if __name__ == "__main__":
    main()
