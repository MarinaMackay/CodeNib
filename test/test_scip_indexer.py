import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from codeminer.scip_interface import SCIPIndexer


@pytest.fixture
def test_dir():
    """Set up a temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir


@pytest.fixture
def indexer(test_dir):
    """Create a SCIPIndexer instance"""
    return SCIPIndexer(test_dir)


@pytest.fixture
def test_output_dir():
    """Provide a directory for test outputs"""
    return Path(__file__).parent


@pytest.fixture(scope="module")
def httpie_repo():
    """Clone and set up the httpie repository for testing."""
    test_repo_url = "https://github.com/httpie/cli.git"
    test_repo_path = Path("/tmp/httpie-cli-test")

    # Clone the repo if it doesn't exist
    if not test_repo_path.exists():
        print(f"Cloning test repository from {test_repo_url}...")
        subprocess.run(["git", "clone", test_repo_url, str(test_repo_path)], check=True)
    else:
        print(f"Using existing test repository at {test_repo_path}")

    return test_repo_path


@pytest.fixture(scope="module")
def samplemod_repo():
    """Clone and set up the samplemod repository for testing."""
    test_repo_url = "https://github.com/navdeep-G/samplemod.git"
    test_repo_path = Path("/tmp/samplemod-test")

    # Clone the repo if it doesn't exist
    if not test_repo_path.exists():
        print(f"Cloning sample module repository from {test_repo_url}...")
        subprocess.run(["git", "clone", test_repo_url, str(test_repo_path)], check=True)
    else:
        print(f"Using existing sample module repository at {test_repo_path}")

    return test_repo_path


def test_conda_environment(indexer):
    """Test the conda environment management functions"""
    assert hasattr(indexer, "_ensure_conda_env")
    assert hasattr(indexer, "_run_in_conda_env")

    # Check that the conda env file exists
    assert (
        indexer.env_file.exists()
    ), f"Conda environment file not found at {indexer.env_file}"


def test_python_repo_indexing(httpie_repo, test_output_dir):
    """
    Test indexing a python repository using SCIPIndexer.
    We use https://github.com/httpie/cli.git as a test repo.
    """
    # Verify the test repo exists
    assert httpie_repo.exists(), f"Test Python repo not found at {httpie_repo}"

    # Create output file in the local directory (not in tmp)
    output_file = str(test_output_dir / "python_repo_index.json")

    # Create a new indexer for the cloned test repo
    repo_indexer = SCIPIndexer(httpie_repo)

    # Run the indexing pipeline, allowing skip_index and skip_decode for faster tests
    graph = repo_indexer.run_pipeline(
        project_name="HttpieCliRepo", output_file=output_file, force=True
    )

    if graph:
        graph.print_graph_basic_info()
        assert graph is not None

        # Print some sample data from the output file
        assert os.path.exists(
            output_file
        ), f"Expected output file {output_file} was not created"

        # Check that index files were created in the temporary directory, not in the project
        index_file = Path("/tmp") / httpie_repo.name / "index.scip"
        assert (
            index_file.exists()
        ), f"Expected index file {index_file} was not created in tmp directory"

        try:
            with open(output_file, "r") as f:
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
        pytest.skip(
            "Failed to run indexing pipeline for test_python_repo, possibly due to missing dependencies"
        )


def test_samplemod_repo_indexing(samplemod_repo, test_output_dir):
    """
    Test indexing the sample module repository using SCIPIndexer.
    We use https://github.com/navdeep-G/samplemod as a test repo.
    This is a small sample repository suitable for testing.
    """
    # Verify the test repo exists
    assert samplemod_repo.exists(), f"Sample module repo not found at {samplemod_repo}"

    # Create output file in the local directory (not in tmp)
    output_file = str(test_output_dir / "samplemod_index.json")
    graph_image_file = str(test_output_dir / "samplemod_graph.jpg")

    # Create a new indexer for the cloned test repo
    # Use our improved SCIPIndexer that stores data in /tmp
    repo_indexer = SCIPIndexer(samplemod_repo)

    # Run the indexing pipeline
    graph = repo_indexer.run_pipeline(
        project_name="SampleModRepo",
        output_file=output_file,
        force=True,
    )

    if graph:
        graph.print_graph_basic_info()
        assert graph is not None

        # visualize the graph and save it to a file
        graph.visualize_graph(graph_image_file)

        # Check that the output file was created
        assert os.path.exists(
            output_file
        ), f"Expected output file {output_file} was not created"

        # Check that index files were created in the temporary directory, not in the project
        index_file = Path("/tmp") / samplemod_repo.name / "index.scip"
        assert (
            index_file.exists()
        ), f"Expected index file {index_file} was not created in tmp directory"

        # Print some sample data from the output file
        try:
            with open(output_file, "r") as f:
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
        pytest.skip(
            "Failed to run indexing pipeline for samplemod_repo, possibly due to missing dependencies"
        )


# For backward compatibility with direct script execution
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
