#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from codeminer.agent.tools import search_code_snippets  # noqa: E402


class TestSearchCodeSnippets(unittest.TestCase):
    """Test cases for the search_code_snippets function."""

    def setUp(self):
        self.repo_path = str(Path(__file__).parent.parent)

    def test_search_file_path(self):
        """Test searching for a specific file."""
        result = search_code_snippets(
            search_terms=["codeminer/search.py"], repo_path=self.repo_path
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0, "Search result should not be empty")
        self.assertIn("search.py", result, "Result should contain file reference")

    def test_search_specific_line_numbers(self):
        """Test retrieving specific line numbers from a file."""
        result = search_code_snippets(
            line_nums=[10, 20],
            file_path_or_pattern="codeminer/search.py",
            repo_path=self.repo_path,
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0, "Search result should not be empty")

    def test_search_with_file_pattern(self):
        """Test searching with file pattern matching."""
        result = search_code_snippets(
            search_terms=["search_code_snippets"],
            file_path_or_pattern="codeminer/agent/*.py",
            repo_path=self.repo_path,
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0, "Search result should not be empty")

    def test_search_multiple_terms(self):
        """Test searching for multiple terms."""
        result = search_code_snippets(
            search_terms=["BM25CodeIndexer", "search"], repo_path=self.repo_path
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0, "Search result should not be empty")

    def test_search_file_entity_function(self):
        """Test searching for a specific function in a file using file_path:entity format."""
        result = search_code_snippets(
            search_terms=["codeminer/agent/tools.py:search_code_snippets"],
            repo_path=self.repo_path,
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0, "Search result should not be empty")
        self.assertIn(
            "search_code_snippets", result, "Result should contain the function name"
        )
        self.assertIn(
            "def search_code_snippets",
            result,
            "Result should contain function definition",
        )
        self.assertIn("Entity:", result, "Result should indicate it's an entity search")
        self.assertNotIn(
            "#!/usr/bin/env python3", result, "Should not return entire file content"
        )

    def test_search_file_entity_class(self):
        """Test searching for a specific class in a file using file_path:entity format."""
        result = search_code_snippets(
            search_terms=["test/test_agent_tools.py:TestSearchCodeSnippets"],
            repo_path=self.repo_path,
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0, "Search result should not be empty")
        self.assertIn(
            "TestSearchCodeSnippets", result, "Result should contain the class name"
        )
        self.assertIn(
            "class TestSearchCodeSnippets",
            result,
            "Result should contain class definition",
        )
        self.assertIn("Entity:", result, "Result should indicate it's an entity search")


if __name__ == "__main__":
    unittest.main(verbosity=2, exit=False)
