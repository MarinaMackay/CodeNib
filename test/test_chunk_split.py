#!/usr/bin/env python3
"""Tests for line-based splitting via max_lines_per_chunk and vector store usage."""

import unittest
from pathlib import Path
from collections import defaultdict, Counter

from codeminer.code_chunker import CodeChunker
from codeminer.embedding.vector_store import create_code_vector_store


class TestChunkSplit(unittest.TestCase):
    def setUp(self):
        self.repo_path = Path(__file__).parent / "simple_repo"
        if not self.repo_path.exists():
            self.skipTest(f"Repo not found: {self.repo_path}")
        
        # Generate chunks for use in both tests
        chunker = CodeChunker(language="python", max_lines_per_chunk=20)
        self.chunks = chunker.chunk_repository(str(self.repo_path), languages=["python"])

    def test_chunk_split_with_duplicate_node_ids(self):
        """Test that splitting creates multiple chunks for the same node_id and print all chunk content."""
        print("\n" + "=" * 80)
        print("CHUNK SPLIT INSPECTION - simple_repo with max_lines_per_chunk=20")
        print("=" * 80)
        print(f"Total chunks: {len(self.chunks)}")

        # Group by file and print details
        by_file = defaultdict(list)
        for ch in self.chunks:
            by_file[ch.file].append(ch)

        for file_path, file_chunks in sorted(by_file.items()):
            rel = Path(file_path).relative_to(self.repo_path)
            print(f"\n-- File: {rel} ({len(file_chunks)} chunks) --")
            for i, ch in enumerate(sorted(file_chunks, key=lambda c: (c.start_line, c.end_line))):
                num_lines = ch.end_line - ch.start_line + 1
                print(
                    f"[{i+1}] type={ch.chunk_type}, name={ch.name}, "
                    f"lines={ch.start_line}-{ch.end_line} ({num_lines} lines)"
                )
                print(f"node_id={ch.node_id}")
                print("CONTENT:")
                print(ch.content)
                print("-" * 60)

        # Verify there exists some duplicate node_id across multiple chunks
        counts = Counter(ch.node_id for ch in self.chunks)
        multi_ids = [nid for nid, cnt in counts.items() if cnt > 1]
        self.assertGreaterEqual(
            len(multi_ids),
            1,
            "Expected at least one symbol/file to be split into multiple chunks",
        )

if __name__ == "__main__":
    unittest.main() 