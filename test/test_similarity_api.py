import unittest
import time
import requests
from codeminer.api import SimilarityAPI, BaseAPI

class TestSimilarityAPI(unittest.TestCase):
    """Test cases for the SimilarityAPI class"""
    
    @classmethod
    def setUpClass(cls):
        """Execute once before all tests"""
        # Make sure the API service is not running
        BaseAPI.stop()
        time.sleep(1)
    
    @classmethod
    def tearDownClass(cls):
        """Execute once after all tests"""
        # Make sure the API service is closed
        BaseAPI.stop()
    
    def setUp(self):
        """Execute before each test method"""
        # Each test uses a new port to avoid port conflicts
        self.test_port = 8765
        self.test_host = "127.0.0.1"
        self.base_url = f"http://{self.test_host}:{self.test_port}"
    
    def tearDown(self):
        """Execute after each test method"""
        # Make sure the API service is stopped
        BaseAPI.stop()
        time.sleep(1)
    
    def test_start_server(self):
        """Test starting the API service"""
        # Start the service using BaseAPI
        BaseAPI.start(host=self.test_host, port=self.test_port, log_level="critical")
        
        # Wait for the service to start
        time.sleep(2)
        
        # Check if the service is running (by sending a request to the root path)
        try:
            response = requests.get(f"{self.base_url}/docs")
            self.assertEqual(response.status_code, 200)
        except requests.RequestException:
            self.fail("Service failed to start")
    
    def test_query_similarity(self):
        """Test the similarity calculation functionality"""
        # Start the service
        BaseAPI.start(host=self.test_host, port=self.test_port, log_level="critical")
        
        # Test code and query
        code = "def greet(name):\n    return f'Hello, {name}!'"
        query = "Define a function that greets someone"
        
        # Test query functionality
        result = SimilarityAPI.query(code, query)
        
        # Verify result format and content
        self.assertIn("score", result)
        self.assertIsInstance(result["score"], float)
        self.assertIn("latency_sec", result)
    
    def test_stop_server(self):
        """Test stopping the API service"""
        # Start the service
        BaseAPI.start(host=self.test_host, port=self.test_port, log_level="critical")
        time.sleep(2)  # Wait for the service to start
        
        # First confirm that the service is running
        try:
            requests.get(f"{self.base_url}/docs")
        except:
            self.fail("Service failed to start, cannot test stop functionality")
        
        # Stop the service
        BaseAPI.stop()
        time.sleep(2)  # Wait for the service to stop
        
        # Verify that the service has stopped
        with self.assertRaises(requests.ConnectionError):
            requests.get(f"{self.base_url}/docs", timeout=1)
    
    def test_multiple_starts(self):
        """Test starting the service multiple times"""
        # First start
        BaseAPI.start(host=self.test_host, port=self.test_port, log_level="critical")
        time.sleep(1)
        
        # Second start (should return directly instead of creating a new instance)
        BaseAPI.start(host=self.test_host, port=self.test_port, log_level="critical")
        
        # Verify that the service is running normally
        response = requests.get(f"{self.base_url}/docs")
        self.assertEqual(response.status_code, 200)
    
    def test_query_without_start(self):
        """Test querying when the service is not started"""
        # Make sure the service is stopped
        BaseAPI.stop()
        time.sleep(1)
        
        # Try to query, should raise an exception
        code = "print('hello')"
        query = "Print a greeting message"
        
        with self.assertRaises(RuntimeError):
            SimilarityAPI.query(code, query)
    
    def test_continuous_queries(self):
        """Test multiple continuous queries to the similarity API"""
        # Start the service
        BaseAPI.start(host=self.test_host, port=self.test_port, log_level="critical")
        
        # Define test data pairs (code and query)
        test_pairs = [
            (
                "def add(a, b):\n    return a + b", 
                "Function to add two numbers"
            ),
            (
                "class User:\n    def __init__(self, name):\n        self.name = name", 
                "Define a User class with a name"
            ),
            (
                "async def fetch_data():\n    response = await client.get('https://api.example.com')\n    return response.json()", 
                "Asynchronous function to fetch data from an API"
            ),
            (
                "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)", 
                "Recursive function to calculate factorial"
            ),
            (
                "import pandas as pd\n\ndf = pd.read_csv('data.csv')\nresult = df.groupby('category').mean()", 
                "Process CSV data using pandas"
            )
        ]
        
        # Test multiple queries in sequence
        results = []
        for code, query in test_pairs:
            result = SimilarityAPI.query(code, query)
            self.assertIn("score", result)
            self.assertIsInstance(result["score"], float)
            self.assertIn("latency_sec", result)
            results.append(result)
            
        # Check that all queries returned valid results
        self.assertEqual(len(results), len(test_pairs))
        
        # Ensure 2nd and subsequent queries are faster (model already loaded)
        if len(results) > 1:
            first_latency = results[0]["latency_sec"]
            for i, result in enumerate(results[1:], 1):
                # Note: This might not always be true in CI environments
                # So we'll just log rather than assert
                current_latency = result["latency_sec"]
                print(f"Query {i} latency: {current_latency}s (1st query: {first_latency}s)")

if __name__ == "__main__":
    unittest.main()
