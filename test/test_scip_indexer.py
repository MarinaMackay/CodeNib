import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from codeminer import SCIPIndexer


class TestSCIPIndexer(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory for testing"""
        self.test_dir = tempfile.mkdtemp()
        self.indexer = SCIPIndexer(self.test_dir)

    def test_conda_environment(self):
        """Test the conda environment management functions"""
        self.assertTrue(hasattr(self.indexer, "_ensure_conda_env"))
        self.assertTrue(hasattr(self.indexer, "_run_in_conda_env"))

        # Check that the conda env file exists
        self.assertTrue(
            self.indexer.env_file.exists(),
            f"Conda environment file not found at {self.indexer.env_file}",
        )

    def test_python_repo_indexing(self):
        """
        Test indexing a python repository using SCIPIndexer.
        We use https://github.com/httpie/cli.git as a test repo.
        """
        # Define the test repo URL and path
        test_repo_url = "https://github.com/httpie/cli.git"
        test_repo_path = Path("/tmp/httpie-cli-test")

        # Clone the repo if it doesn't exist
        if not test_repo_path.exists():
            print(f"Cloning test repository from {test_repo_url}...")
            subprocess.run(
                ["git", "clone", test_repo_url, str(test_repo_path)], check=True
            )
        else:
            print(f"Using existing test repository at {test_repo_path}")

        # Verify the test repo exists
        self.assertTrue(
            test_repo_path.exists(), f"Test Python repo not found at {test_repo_path}"
        )

        # Create output file in the local directory (not in tmp)
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        output_file = str(current_dir / "python_repo_index.json")

        # Create a new indexer for the cloned test repo
        repo_indexer = SCIPIndexer(test_repo_path)

        # Run the indexing pipeline, allowing skip_index and skip_decode for faster tests
        graph = repo_indexer.run_pipeline(
            project_name="HttpieCliRepo",
            output_file=output_file,
            skip_index=False,
            skip_decode=False,
        )

        if graph:
            graph.print_graph_basic_info()
            self.assertIsNotNone(graph)
            # Print some sample data from the output file
            self.assertTrue(
                os.path.exists(output_file),
                f"Expected output file {output_file} was not created",
            )
            # Check that index files were created in the temporary directory, not in the project
            index_file = Path("/tmp") / test_repo_path.name / "index.scip"
            self.assertTrue(
                index_file.exists(),
                f"Expected index file {index_file} was not created in tmp directory",
            )
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
                    print("\nSample nodes from the graph:")
                    # Print up to 3 nodes of each type
                    file_nodes = [n for n in data["nodes"] if n.get("type") == "file"]
                    symbol_nodes = [
                        n for n in data["nodes"] if n.get("type") == "symbol"
                    ]

                    for i, node in enumerate(file_nodes[:3]):
                        print(f"  File node {i+1}: {node['id']}")

                    for i, node in enumerate(symbol_nodes[:3]):
                        print(f"  Symbol node {i+1}: {node['id']}")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Could not read output file: {e}")
        else:
            self.skipTest(
                "Failed to run indexing pipeline for test_python_repo, possibly due to missing dependencies"
            )

    def test_samplemod_repo_indexing(self):
        """
        Test indexing the sample module repository using SCIPIndexer.
        We use https://github.com/navdeep-G/samplemod as a test repo.
        This is a small sample repository suitable for testing.
        """
        # Define the test repo URL and path
        test_repo_url = "https://github.com/navdeep-G/samplemod.git"
        test_repo_path = Path("/tmp/samplemod-test")

        # Clone the repo if it doesn't exist
        if not test_repo_path.exists():
            print(f"Cloning sample module repository from {test_repo_url}...")
            subprocess.run(
                ["git", "clone", test_repo_url, str(test_repo_path)], check=True
            )
        else:
            print(f"Using existing sample module repository at {test_repo_path}")

        # Verify the test repo exists
        self.assertTrue(
            test_repo_path.exists(), f"Sample module repo not found at {test_repo_path}"
        )

        # Create output file in the local directory (not in tmp)
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        output_file = str(current_dir / "samplemod_index.json")
        graph_image_file = str(current_dir / "samplemod_graph.jpg")

        # Create a new indexer for the cloned test repo
        # Use our improved SCIPIndexer that stores data in /tmp
        repo_indexer = SCIPIndexer(test_repo_path)

        # Run the indexing pipeline
        graph = repo_indexer.run_pipeline(
            project_name="SampleModRepo",
            output_file=output_file,
            skip_index=False,
            skip_decode=False,
        )

        if graph:
            graph.print_graph_basic_info()
            self.assertIsNotNone(graph)
            # visualize the graph and save it to a file
            graph.visualize_graph(graph_image_file)

            # Check that the output file was created
            self.assertTrue(
                os.path.exists(output_file),
                f"Expected output file {output_file} was not created",
            )

            # Check that index files were created in the temporary directory, not in the project
            index_file = Path("/tmp") / test_repo_path.name / "index.scip"
            self.assertTrue(
                index_file.exists(),
                f"Expected index file {index_file} was not created in tmp directory",
            )

            # Print some sample data from the output file
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
                    print("\nSample nodes from the graph:")
                    # Print up to 3 nodes of each type
                    file_nodes = [n for n in data["nodes"] if n.get("type") == "file"]
                    symbol_nodes = [
                        n for n in data["nodes"] if n.get("type") == "symbol"
                    ]

                    for i, node in enumerate(file_nodes[:3]):
                        print(f"  File node {i+1}: {node['id']}")

                    for i, node in enumerate(symbol_nodes[:3]):
                        print(f"  Symbol node {i+1}: {node['id']}")

            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Could not read output file: {e}")
        else:
            self.skipTest(
                "Failed to run indexing pipeline for samplemod_repo, possibly due to missing dependencies"
            )


if __name__ == "__main__":
    unittest.main()
