#!/usr/bin/env python3
"""Test to check if all Dense chunk node_ids exist in BM25 graph nodes."""

import unittest
from pathlib import Path

from codeminer.bm25_index import BM25CodeIndexer
from codeminer.embedding.vector_store import create_code_vector_store
from codeminer.scip_interface import SCIPIndexer
from codeminer.code_chunker import CodeChunker


class TestDenseCompatibility(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.repo_path = Path(__file__).parent / "simple_repo"
        self.output_path = Path.home() / ".codeminer" / "simple_repo_nodes_test"
        
        # Ensure the test repo exists
        if not self.repo_path.exists():
            self.skipTest(f"Test repository not found at {self.repo_path}")
        
        print(f"Testing with repository: {self.repo_path}")

    def test_dense_compatibility(self):
        """Test that all Dense chunk node_ids exist in BM25 graph nodes"""
        print("\n" + "=" * 80)
        print("DENSE COMPATIBILITY TEST")
        print("=" * 80)
        
        # === Create BM25 nodes ===
        print("\n1. Creating BM25 nodes...")
        try:
            repo_indexer = SCIPIndexer(self.repo_path, output_dir=self.output_path)
            graph = repo_indexer.run_pipeline(project_name="simple_repo_compat", force=True)
            
            if not graph:
                self.fail("Failed to create BM25 graph")
                
            bm25_indexer = BM25CodeIndexer(code_graph=graph)
            
            # Extract all BM25 node IDs from documents
            bm25_node_ids = set()
            for doc in bm25_indexer.documents:
                node_id = doc.metadata.get('node_id', '')
                if node_id:  # Skip empty node_ids
                    bm25_node_ids.add(node_id)
            
            print(f"   Found {len(bm25_node_ids)} BM25 node IDs")
            print(f"   BM25 node IDs: {sorted(bm25_node_ids)}")
            
        except Exception as e:
            self.fail(f"BM25 nodes creation failed: {e}")

        # === Create Dense chunks ===
        print("\n2. Creating Dense chunks...")
        try:
            code_chunker = CodeChunker(language="python", max_lines_per_chunk=50)
            chunks = code_chunker.chunk_repository(str(self.repo_path), languages=["python"])
            
            # Extract all Dense chunk node IDs
            dense_node_ids = set()
            for chunk in chunks:
                if chunk.node_id:  # Skip empty node_ids
                    dense_node_ids.add(chunk.node_id)
            
            print(f"   Found {len(dense_node_ids)} Dense node IDs")
            print(f"   Dense node IDs: {sorted(dense_node_ids)}")
            
        except Exception as e:
            self.fail(f"Dense chunks creation failed: {e}")

        # === Compatibility Check ===
        print("\n3. Compatibility Analysis:")
        print("-" * 50)
        
        # Check which Dense node IDs exist in BM25
        compatible_ids = dense_node_ids.intersection(bm25_node_ids)
        missing_ids = dense_node_ids - bm25_node_ids
        extra_bm25_ids = bm25_node_ids - dense_node_ids
        
        print(f"✅ Compatible node IDs ({len(compatible_ids)}):")
        for node_id in sorted(compatible_ids):
            print(f"   - {node_id}")
            
        if missing_ids:
            print(f"\n❌ Dense node IDs missing in BM25 ({len(missing_ids)}):")
            for node_id in sorted(missing_ids):
                print(f"   - {node_id}")
        else:
            print(f"\n✅ All Dense node IDs found in BM25!")
            
        if extra_bm25_ids:
            print(f"\n📝 BM25-only node IDs ({len(extra_bm25_ids)}):")
            for node_id in sorted(extra_bm25_ids):
                print(f"   - {node_id}")

        # === Results Summary ===
        print("\n" + "=" * 50)
        print("COMPATIBILITY SUMMARY")
        print("=" * 50)
        print(f"Dense chunks: {len(chunks)}")
        print(f"Dense node IDs: {len(dense_node_ids)}")
        print(f"BM25 node IDs: {len(bm25_node_ids)}")
        print(f"Compatible: {len(compatible_ids)}")
        print(f"Missing in BM25: {len(missing_ids)}")
        
        compatibility_rate = len(compatible_ids) / len(dense_node_ids) * 100 if dense_node_ids else 0
        print(f"Compatibility rate: {compatibility_rate:.1f}%")
        
        # Assert that all Dense node IDs exist in BM25
        self.assertEqual(len(missing_ids), 0, 
                        f"Found {len(missing_ids)} Dense node IDs missing in BM25: {missing_ids}")
        
        print("\n✅ COMPATIBILITY TEST PASSED!")
        print("=" * 80)


if __name__ == "__main__":
    unittest.main() 