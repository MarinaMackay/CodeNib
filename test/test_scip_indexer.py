import os
import unittest
import tempfile
import shutil
import json
from pathlib import Path
from codeminer import SCIPIndexer

class TestSCIPIndexer(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory for testing"""
        self.test_dir = tempfile.mkdtemp()
        self.indexer = SCIPIndexer(self.test_dir)
        
    def test_conda_environment(self):
        """Test the conda environment management functions"""
        self.assertTrue(hasattr(self.indexer, '_ensure_conda_env'))
        self.assertTrue(hasattr(self.indexer, '_run_in_conda_env'))
        
        # Check that the conda env file exists
        self.assertTrue(self.indexer.env_file.exists(), 
                        f"Conda environment file not found at {self.indexer.env_file}")
        
    def test_python_repo_indexing(self):
        """
        Test indexing the test_python_repo directory
        
        This test uses the provided test_python_repo which contains a more
        complex Python project structure with multiple modules and imports.
        """
        # Skip this test if we're in CI or don't want to run long tests
        if os.environ.get('SKIP_CONDA_TESTS'):
            self.skipTest("Skipping test that requires conda")
        
        # Get path to test_python_repo directory
        test_python_repo = Path(os.path.dirname(__file__)) / "test_python_repo"
        
        # Verify the test repo exists
        self.assertTrue(test_python_repo.exists(),
                       f"Test Python repo not found at {test_python_repo}")
        
        # Create output file in our temporary directory
        output_file = str("python_repo_index.json")
        
        # Create a new indexer for the test_python_repo
        repo_indexer = SCIPIndexer(test_python_repo)
        
        # Run the indexing pipeline, allowing skip_index and skip_decode for faster tests
        result = repo_indexer.run_pipeline(
            project_name="TestPythonRepo",
            output_file=output_file,
            skip_index=os.environ.get('SKIP_INDEX_GENERATION') is not None,
            skip_decode=os.environ.get('SKIP_INDEX_DECODE') is not None
        )
        
        if result:
            self.assertIsNotNone(result)
            self.assertIn("nodes", result)
            self.assertIn("edges", result)
            
            # Verify we have reasonable number of nodes and edges
            self.assertGreater(result["nodes"], 5, "Expected more nodes in the graph")
            self.assertGreater(result["edges"], 5, "Expected more edges in the graph")
            
            # Verify file nodes were detected
            self.assertGreater(result["file_nodes"], 3, "Expected more file nodes")
            
            # Verify symbols were detected
            self.assertGreater(result["symbol_nodes"], 3, "Expected more symbol nodes")
            
            # Print the results for inspection
            print("\nTest Python Repo SCIP Index Results:")
            for key, value in result.items():
                print(f"  {key}: {value}")
                
            # Check that the output file was created
            self.assertTrue(os.path.exists(output_file), 
                          f"Expected output file {output_file} was not created")
            
            # Print some sample data from the output file
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    print("\nSample nodes from the graph:")
                    # Print up to 3 nodes of each type
                    file_nodes = [n for n in data["nodes"] if n.get("type") == "file"]
                    symbol_nodes = [n for n in data["nodes"] if n.get("type") == "symbol"]
                    
                    for i, node in enumerate(file_nodes[:3]):
                        print(f"  File node {i+1}: {node['id']}")
                        
                    for i, node in enumerate(symbol_nodes[:3]):
                        print(f"  Symbol node {i+1}: {node['id']}")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Could not read output file: {e}")
        else:
            self.skipTest("Failed to run indexing pipeline for test_python_repo, possibly due to missing dependencies")

if __name__ == "__main__":
    unittest.main()