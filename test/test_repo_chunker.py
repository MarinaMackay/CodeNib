#!/usr/bin/env python3
"""
Test script for simple repository code chunking functionality.
"""

from pathlib import Path

from codeminer.code_chunker import CodeChunker


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


def main():
    """Main test function."""
    print("Testing simple repo chunker functionality...\n")
    test_simple_repo_chunker()
    print("\n🎉 Test completed!")


if __name__ == "__main__":
    main()
