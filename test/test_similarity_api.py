import unittest
import time

from fastapi.testclient import TestClient
from codeminer.similarity_api import app


class TestSimilarityAPI(unittest.TestCase):

    def test_similarity_endpoint(self):
        """Test the /similarity endpoint with a code–query pair"""
        client = TestClient(app)
        response = client.post("/similarity", json={
            "code": "def multiply(a, b): return a * b",
            "query": "function to multiply two numbers"
        })

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("score", data)
        self.assertIsInstance(data["score"], float)
        self.assertIn("latency_sec", data)

        print("Score:", data["score"])
        print("Latency:", data["latency_sec"], "sec")

    def test_multiple_requests_latency(self):
        """Test latency of multiple consecutive requests"""
        client = TestClient(app)
        
        # First request (warmup)
        first_response = client.post("/similarity", json={
            "code": "def add(a, b): return a + b",
            "query": "function to add two numbers"
        })
        self.assertEqual(first_response.status_code, 200)
        first_latency = first_response.json()["latency_sec"]
        print(f"\nFirst request latency: {first_latency} seconds (includes model loading time)")
        
        # Multiple subsequent requests
        test_cases = [
            {"code": "def subtract(a, b): return a - b", 
             "query": "function to subtract numbers"},

            {"code": "def divide(a, b): return a / b", 
             "query": "function to divide two values"},

            {"code": "def square(x): return x * x", 
             "query": "function that calculates the square"},

            {"code": "def is_even(n): return n % 2 == 0", 
             "query": "check if number is even"},

            {"code": """
                def process_large_dataset(data):
                    # Initialize results dictionary
                    results = {}
                    
                    # Process each record
                    for record in data:
                        # Extract fields
                        id = record.get('id')
                        values = record.get('values', [])
                        metadata = record.get('metadata', {})
                        
                        # Perform calculations
                        total = sum(values)
                        average = total / len(values) if values else 0
                        max_value = max(values) if values else None
                        min_value = min(values) if values else None
                        
                        # Apply transformations
                        transformed = [v * 2 for v in values]
                        filtered = [v for v in values if v > average]
                        
                        # Store results
                        results[id] = {
                            'total': total,
                            'average': average,
                            'max': max_value,
                            'min': min_value,
                            'transformed': transformed,
                            'filtered': filtered,
                            'metadata': metadata
                        }
                    
                    return results
                """, 
            "query": "A complex data processing function that handles large datasets by calculating various statistics like sum, average, max and min values, performs data transformations and filtering, and organizes results in a structured format"}
        ]
        
        latencies = []
        for i, test_case in enumerate(test_cases):
            start = time.time()
            response = client.post("/similarity", json=test_case)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            latencies.append(data["latency_sec"])
            print(f"Request {i+1} - Similarity: {data['score']:.4f}, Latency: {data['latency_sec']} seconds")
        
        avg_latency = sum(latencies) / len(latencies)
        print(f"\nAverage latency of subsequent requests: {avg_latency:.4f} seconds")
        print(f"Ratio of first request to average subsequent request: {first_latency/avg_latency:.2f}x")


if __name__ == "__main__":
    unittest.main()
