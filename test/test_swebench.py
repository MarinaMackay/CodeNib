import os
import subprocess
import unittest
from pathlib import Path

# Import HuggingFace datasets for direct loading
from datasets import load_dataset

from codeminer.env.process_data import process_swebench_instance


class TestSWEBench(unittest.TestCase):
    """Test cases for SWE-bench dataset processing"""

    @classmethod
    def setUpClass(cls):
        # Load the dataset directly using HuggingFace datasets API
        try:
            print("Loading SWE-bench Lite dataset from Hugging Face...")
            cls.dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
            print(
                f"Successfully loaded SWE-bench Lite dataset with {len(cls.dataset)} instances"
            )
        except Exception as e:
            print(f"Failed to load SWE-bench dataset from Hugging Face: {e}")

    def setUp(self):
        """Set up test environment for each test case"""
        # Use ./codeminer as the cache directory
        user_home = str(Path.home())
        self.cache_dir = os.path.join(user_home, "codeminer")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load test data - first try the class dataset, then the cached file
        if hasattr(self.__class__, "dataset"):
            self.test_data = [item for item in self.__class__.dataset]
        else:
            self.skipTest(f"Could not load test data")

    def test_process_swebench_instance(self):
        """Test processing a real SWE-bench instance"""
        # Take the first instance from the test dataset
        instance = self.dataset[0]
        print(
            f"Testing with instance {instance['instance_id']} from repo {instance['repo']}"
        )

        # Process the instance
        try:
            repo_path = process_swebench_instance(instance, self.cache_dir)
            print(f"Repository processed at path: {repo_path}")

            # Check that the repo directory was created
            self.assertTrue(os.path.exists(repo_path))

            # Check that the .git directory exists
            git_dir = os.path.join(repo_path, ".git")
            self.assertTrue(os.path.exists(git_dir))

            # Verify we're at the right commit
            current_dir = os.getcwd()
            os.chdir(repo_path)

            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                commit_hash = result.stdout.strip()
                print(f"Checked out commit hash: {commit_hash}")

                # Check that we're at the expected commit
                self.assertEqual(commit_hash, instance["base_commit"])

            finally:
                os.chdir(current_dir)

        except Exception as e:
            self.fail(f"Error processing SWE-bench instance: {e}")


if __name__ == "__main__":
    unittest.main()
