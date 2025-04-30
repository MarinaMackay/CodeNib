import os
import subprocess
from typing import Any, Dict

from ..log_utils import get_logger

logger = get_logger(__name__)


def process_swebench_instance(
    dataset_row: Dict[str, Any], cache_dir: str = "~/.codeminer"
) -> str:
    """
    Process a dataset instance by:
    1. Downloading the repository if not exists
    2. Checking out the specific commit

    Args:
        dataset_row: A row from the SWE-bench dataset containing repo_name, instance_id, and base_commit
        cache_dir: Directory to store repositories

    Returns:
        Path to the checked out repository
    """
    # Extract relevant information from the dataset row
    repo_name = dataset_row["repo"]
    base_commit = dataset_row["base_commit"]

    # Create cache directory if it doesn't exist
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # Repository paths
    repo_dir_name = repo_name.replace("/", "_")
    repo_path = os.path.join(cache_dir, repo_dir_name)

    # Check if repo exists
    if not os.path.exists(repo_path):
        logger.info(f"Downloading repository {repo_name} to {repo_path}")

        # Clone the repository
        git_url = f"https://github.com/{repo_name}.git"
        try:
            subprocess.run(
                ["git", "clone", git_url, repo_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            logger.error(f"STDERR: {e.stderr.decode('utf-8')}")
            raise RuntimeError(f"Failed to clone repository {repo_name}")
    else:
        logger.info(f"Repository {repo_name} already exists at {repo_path}")

    # Change to the repository directory
    original_dir = os.getcwd()
    os.chdir(repo_path)

    try:
        # Fetch all updates to ensure we can checkout the commit
        logger.info("Fetching updates from remote repository")
        subprocess.run(
            ["git", "fetch", "--all"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Checkout to the base commit
        logger.info(f"Checking out commit {base_commit}")
        try:
            subprocess.run(
                ["git", "checkout", base_commit],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to checkout commit {base_commit}: {e}")
            logger.error(f"STDERR: {e.stderr.decode('utf-8')}")
            raise RuntimeError(
                f"Failed to checkout commit {base_commit} for repo {repo_name}"
            )

        logger.info(f"Successfully checked out {repo_name} at commit {base_commit}")

    finally:
        # Return to original directory
        os.chdir(original_dir)

    return repo_path
