import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from codeminer.env.process_data import process_swebench_instance

# Add parent directory to path to import codeminer modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSWEBench(unittest.TestCase):
    """Test cases for SWE-bench dataset processing"""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment by ensuring the SWE-bench data exists"""
        # Define the benchmark directory and data paths
        cls.benchmark_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark"
        )
        cls.data_dir = os.path.join(cls.benchmark_dir, "swebench_lite_data")
        cls.test_data_path = os.path.join(cls.data_dir, "test.json")

        # Check if the data file exists
        if not os.path.exists(cls.test_data_path):
            print("SWE-bench test data not found. Downloading...")
            # Run the script to download the data
            script_path = os.path.join(cls.benchmark_dir, "swebench_lite.py")
            try:
                subprocess.run(
                    [
                        sys.executable,
                        script_path,
                        "--output-dir",
                        cls.data_dir,
                        "--split",
                        "test",
                    ],
                    check=True,
                )
                print(f"Downloaded SWE-bench test data to {cls.test_data_path}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to download SWE-bench data: {e}")
                raise

    def setUp(self):
        """Set up test environment for each test case"""
        # Use ./codeminer as the cache directory
        user_home = str(Path.home())
        self.cache_dir = os.path.join(user_home, "codeminer")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load test data
        try:
            with open(self.__class__.test_data_path, "r") as f:
                self.test_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.skipTest(f"Could not load test data: {e}")

    def test_process_swebench_instance(self):
        """Test processing a real SWE-bench instance"""
        # Skip if no data is available
        if not hasattr(self, "test_data") or not self.test_data:
            self.skipTest("No test data available")

        # Take the first instance from the test dataset
        instance = self.test_data[0]

        # Process the instance
        try:
            repo_path = process_swebench_instance(instance, self.cache_dir)

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

                # Check that we're at the expected commit
                self.assertEqual(commit_hash, instance["base_commit"])

            finally:
                os.chdir(current_dir)

        except Exception as e:
            self.fail(f"Error processing SWE-bench instance: {e}")


if __name__ == "__main__":
    unittest.main()
