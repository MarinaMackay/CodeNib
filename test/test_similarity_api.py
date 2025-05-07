import unittest
import time
import pytest
import requests
from codeminer import SimilarityAPI

class TestSimilarityAPI(unittest.TestCase):
    """Test functionality of the SimilarityAPI class"""
    
    @classmethod
    def setUpClass(cls):
        """Execute once before all tests"""
        # Make sure the API service is not running
        SimilarityAPI.stop()
        time.sleep(1)  # Wait for the service to completely stop
    
    @classmethod
    def tearDownClass(cls):
        """Execute once after all tests"""
        # Make sure the API service is closed
        SimilarityAPI.stop()
    
    def setUp(self):
        """Execute before each test method"""
        # Each test uses a new port to avoid port conflicts
        self.test_port = 8765
        self.test_host = "127.0.0.1"
        self.base_url = f"http://{self.test_host}:{self.test_port}"
    
    def tearDown(self):
        """Execute after each test method"""
        # Make sure the API service is stopped
        SimilarityAPI.stop()
        time.sleep(1)  # Wait for the service to completely stop
    
    def test_start_server(self):
        """Test starting the API service"""
        # Start the service
        SimilarityAPI.start(host=self.test_host, port=self.test_port, log_level="critical")
        
        # Wait for the service to start
        time.sleep(2)
        
        # Check if the service is running (by sending a request to the root path)
        try:
            response = requests.get(f"{self.base_url}/docs")
            self.assertEqual(response.status_code, 200)
            print("Service started successfully")
        except requests.RequestException:
            self.fail("Service failed to start")
    
    def test_query_similarity(self):
        """Test the similarity calculation functionality"""
        # Start the service
        SimilarityAPI.start(host=self.test_host, port=self.test_port, log_level="critical")
        
        # Test code and query
        code = "def greet(name):\n    return f'Hello, {name}!'"
        query = "Define a function that greets someone"
        
        # Test query functionality
        result = SimilarityAPI.query(code, query)
        
        # Verify result format and content
        self.assertIn("score", result)
        self.assertIsInstance(result["score"], float)
        self.assertIn("latency_sec", result)
        
        # Print results
        print(f"Similarity score: {result['score']}")
        print(f"Latency: {result['latency_sec']} seconds")
    
    def test_stop_server(self):
        """Test stopping the API service"""
        # Start the service
        SimilarityAPI.start(host=self.test_host, port=self.test_port, log_level="critical")
        time.sleep(2)  # Wait for the service to start
        
        # First confirm that the service is running
        try:
            requests.get(f"{self.base_url}/docs")
        except:
            self.fail("Service failed to start, cannot test stop functionality")
        
        # Stop the service
        SimilarityAPI.stop()
        time.sleep(2)  # Wait for the service to stop
        
        # Verify that the service has stopped
        with self.assertRaises(requests.ConnectionError):
            requests.get(f"{self.base_url}/docs", timeout=1)
        print("Service stopped successfully")
    
    def test_multiple_starts(self):
        """Test starting the service multiple times"""
        # First start
        SimilarityAPI.start(host=self.test_host, port=self.test_port, log_level="critical")
        time.sleep(1)
        
        # Second start (should return directly instead of creating a new instance)
        SimilarityAPI.start(host=self.test_host, port=self.test_port, log_level="critical")
        
        # Verify that the service is running normally
        response = requests.get(f"{self.base_url}/docs")
        self.assertEqual(response.status_code, 200)
        print("Multiple starts test passed")
    
    def test_query_without_start(self):
        """Test querying when the service is not started"""
        # Make sure the service is stopped
        SimilarityAPI.stop()
        time.sleep(1)
        
        # Try to query, should raise an exception
        code = "print('hello')"
        query = "Print a greeting message"
        
        with self.assertRaises(RuntimeError):
            SimilarityAPI.query(code, query)
        print("Query without start test passed")

if __name__ == "__main__":
    unittest.main()
