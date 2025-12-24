import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Union

import datasets
from datasets import Features, Sequence, Value

from ..log_utils import get_logger

logger = get_logger(__name__)


def load_filter_swebench_dataset(
    args: argparse.Namespace,
) -> datasets.arrow_dataset.Dataset:
    ret = load_filter_swebench_dataset_explicit(
        dataset=args.dataset, filter_instance=args.filter_instance, split=args.split
    )
    # Cannot has both idx_list and idx_range
    assert not (
        hasattr(args, "idx_list") and hasattr(args, "idx_range")
    ), "Cannot has both idx_list and idx_range in arguments"
    if hasattr(args, "idx_list"):
        if args.filter_instance != ".*":
            logger.info(
                (
                    "Running idx_list on a filtered (non-full) dataset."
                    "Please make sure this is expected."
                )
            )
        return ret.select(args.idx_list)
    elif hasattr(args, "idx_range"):
        if args.filter_instance != ".*":
            logger.info(
                (
                    "Running idx_range on a filtered (non-full) dataset."
                    "Please make sure this is expected."
                )
            )
        start_idx = args.idx_range[0]
        end_idx = args.idx_range[1]
        assert start_idx < end_idx, "start_idx should be smaller than end_idx"
        return ret.select(range(start_idx, end_idx))
    else:
        return ret


def load_filter_swebench_dataset_explicit(
    dataset: str, filter_instance: str, split: str
) -> datasets.arrow_dataset.Dataset:
    cache_dir = str(Path.home()) + "/.codeminer"
    # Create cache directory if it doesn't exist
    cache_dir = os.path.abspath(cache_dir)
    if not os.path.exists(cache_dir):
        logger.info(f"Creating cache directory at {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)
    dataset_file = f'{dataset.replace("/", "__")}_{split}.json'
    dataset_path = f"{cache_dir}/{dataset_file}"
    if not os.path.exists(dataset_path):
        ds = datasets.load_dataset(dataset, split=split)
        logger.info(f"Loaded {len(ds)} instances from {dataset} dataset, split {split}")
        ds.to_json(dataset_path)
    else:
        logger.info(f"Dataset already exists at {dataset_path}")
        data_files = {split: dataset_path}

        # Define base features common to all SWE-bench variants
        base_features = {
            "repo": Value("string"),
            "instance_id": Value("string"),
            "base_commit": Value("string"),
            "patch": Value("string"),
            "test_patch": Value("string"),
            "problem_statement": Value("string"),
            "hints_text": Value("string"),
            "created_at": Value("string"),
            "version": Value("string"),
        }

        # Different datasets have different schemas for FAIL_TO_PASS and PASS_TO_PASS
        if "multilingual" in dataset.lower():
            # SWE-bench Multilingual uses list type for test fields
            base_features["FAIL_TO_PASS"] = Sequence(Value("string"))
            base_features["PASS_TO_PASS"] = Sequence(Value("string"))
        else:
            # SWE-bench Verified and original use string type
            base_features["FAIL_TO_PASS"] = Value("string")
            base_features["PASS_TO_PASS"] = Value("string")
            base_features["environment_setup_commit"] = Value("string")

        # Add difficulty field for SWE-bench Verified
        if "verified" in dataset.lower():
            base_features["difficulty"] = Value("string")

        ft = Features(base_features)
        ds = datasets.load_dataset(
            "json", data_files=data_files, split=split, features=ft
        )
        logger.info(f"Loaded {len(ds)} instances from cached dataset at {dataset_path}")
    return ds.filter(
        input_columns=["instance_id"],
        function=lambda x: bool(re.match(filter_instance, x)),
    )


def process_swebench_instance(
    dataset_row: Dict[str, Any], cache_dir: Union[Path, str] = "~/.codeminer"
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

    # Properly expand and normalize cache_dir
    if isinstance(cache_dir, str):
        cache_dir = os.path.expanduser(cache_dir)
    cache_dir = str(Path(cache_dir).absolute())
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
