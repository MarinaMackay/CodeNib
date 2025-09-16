import unittest
import time
from codeminer.api import SimilarityAPI, BaseAPI

class TestSimilarityAPI(unittest.TestCase):
    """Test cases for the SimilarityAPI class"""
    
    @classmethod
    def setUpClass(cls):
        """Setup before all tests"""
        # Ensure API service is not running
        BaseAPI.stop()
        time.sleep(1)
        
        # Define large code blocks for testing
        cls.large_code_blocks = [
            """
def fibonacci_generator(n):
    '''Generate the first n terms of Fibonacci sequence'''
    a, b = 0, 1
    for i in range(n):
        yield a
        a, b = b, a + b

class DataProcessor:
    def __init__(self, data_source):
        self.data_source = data_source
        self.processed_data = []
    
    def load_data(self):
        '''Load data from data source'''
        try:
            with open(self.data_source, 'r') as file:
                raw_data = file.read()
                return self.parse_data(raw_data)
        except FileNotFoundError:
            print(f"File {self.data_source} not found")
            return None
    
    def parse_data(self, raw_data):
        '''Parse raw data'''
        lines = raw_data.strip().split('\\n')
        parsed = []
        for line in lines:
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 2:
                    parsed.append({
                        'id': parts[0].strip(),
                        'value': float(parts[1].strip()) if parts[1].strip().replace('.', '').isdigit() else 0
                    })
        return parsed
            """,
            
            """
import asyncio
import aiohttp
from typing import List, Dict, Optional
import json

class AsyncAPIClient:
    '''Async API client with batch requests and retry mechanism'''
    
    def __init__(self, base_url: str, timeout: int = 30, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        '''Send HTTP request with retry mechanism'''
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"Request failed with status: {response.status}")
                        
            except asyncio.TimeoutError:
                print(f"Request timeout, retry {attempt + 1}/{self.max_retries}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                print(f"Request exception: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(1)
        
        return None
    
    async def batch_requests(self, requests: List[Dict]) -> List[Optional[Dict]]:
        '''Send batch requests'''
        tasks = []
        for req in requests:
            task = self.make_request(
                req.get('method', 'GET'),
                req['endpoint'],
                json=req.get('data'),
                params=req.get('params')
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
            """,
            
            """
from dataclasses import dataclass
from typing import Union, List, Optional, Callable
import numpy as np
from abc import ABC, abstractmethod

@dataclass
class Point:
    x: float
    y: float
    
    def distance_to(self, other: 'Point') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Shape(ABC):
    '''Abstract shape base class'''
    
    @abstractmethod
    def area(self) -> float:
        pass
    
    @abstractmethod
    def perimeter(self) -> float:
        pass
    
    @abstractmethod
    def contains_point(self, point: Point) -> bool:
        pass

class Circle(Shape):
    def __init__(self, center: Point, radius: float):
        self.center = center
        self.radius = radius
    
    def area(self) -> float:
        return np.pi * self.radius ** 2
    
    def perimeter(self) -> float:
        return 2 * np.pi * self.radius
    
    def contains_point(self, point: Point) -> bool:
        return self.center.distance_to(point) <= self.radius

class Rectangle(Shape):
    def __init__(self, top_left: Point, width: float, height: float):
        self.top_left = top_left
        self.width = width
        self.height = height
    
    def area(self) -> float:
        return self.width * self.height
    
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)
    
    def contains_point(self, point: Point) -> bool:
        return (self.top_left.x <= point.x <= self.top_left.x + self.width and
                self.top_left.y <= point.y <= self.top_left.y + self.height)

class GeometryCalculator:
    '''Geometry calculation utility class'''
    
    @staticmethod
    def calculate_total_area(shapes: List[Shape]) -> float:
        return sum(shape.area() for shape in shapes)
    
    @staticmethod
    def find_shapes_containing_point(shapes: List[Shape], point: Point) -> List[Shape]:
        return [shape for shape in shapes if shape.contains_point(point)]
    
    @staticmethod
    def apply_transformation(shapes: List[Shape], transform_func: Callable[[Shape], Shape]) -> List[Shape]:
        return [transform_func(shape) for shape in shapes]
            """
        ]
        
        cls.queries = [
            "Bug: Fibonacci generator function returns incorrect sequence after the 10th iteration. The function seems to lose precision when dealing with large numbers and sometimes returns negative values. Users report that when calling fibonacci_generator(50), the sequence becomes corrupted around the 45th element. The issue appears to be related to integer overflow or incorrect variable assignment in the loop logic.",
            "Bug: AsyncAPIClient fails to handle connection timeouts properly during batch requests. When processing multiple concurrent requests, the client sometimes hangs indefinitely instead of applying the retry mechanism. Users experience this issue particularly when the server is under heavy load. The session management seems problematic and may not be properly closing connections, leading to resource leaks and eventual application crashes.",
            "Bug: GeometryCalculator produces incorrect area calculations for overlapping shapes. When shapes overlap, the calculate_total_area method double-counts the intersection areas instead of computing the union. Additionally, the contains_point method for Rectangle class has boundary condition errors - points exactly on the border are inconsistently classified as inside or outside the shape, causing issues in collision detection algorithms."
        ]
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup after all tests"""
        BaseAPI.stop()
    
    def test_1_import_and_startup(self):
        """Test 1: Import SimilarityAPI, test BaseAPI startup and config query"""
        print("\n=== Test 1: API Startup and Configuration Query ===")
        
        # Start API server
        BaseAPI.start(host="127.0.0.1", port=8765, log_level="critical")
        
        # Verify server is started
        self.assertTrue(BaseAPI._initialized, "API server should be started")
        
        # Test configuration query (get current config via configure method)
        try:
            config_result = SimilarityAPI.configure()
            print(f"Current configuration: {config_result}")
            self.assertIn("status", config_result)
            self.assertEqual(config_result["status"], "success")
        except Exception as e:
            self.fail(f"Configuration query failed: {e}")
    
    def test_2_continuous_queries_default(self):
        """Test 2: Continuous queries with default configuration on large code blocks"""
        print("\n=== Test 2: Continuous Queries with Default Configuration ===")
        
        results = []
        for i, (code, query) in enumerate(zip(self.large_code_blocks, self.queries)):
            print(f"Query {i+1}: Bug description length: {len(query)} chars")
            result = SimilarityAPI.query(code, query)
            
            # Verify result format
            self.assertIn("score", result)
            self.assertIn("latency_sec", result)
            self.assertIsInstance(result["score"], float)
            
            print(f"  Similarity score: {result['score']:.4f}, Latency: {result['latency_sec']:.4f}s")
            results.append(result)
        
        self.assertEqual(len(results), len(self.large_code_blocks))
    
    def test_3_auto_mode_queries(self):
        """Test 3: Switch to auto mode and repeat queries"""
        print("\n=== Test 3: Queries in Auto Mode ===")
        
        # Switch to auto mode
        config_result = SimilarityAPI.configure(device="auto")
        print(f"Switched to auto mode: {config_result}")
        
        results = []
        for i, (code, query) in enumerate(zip(self.large_code_blocks, self.queries)):
            print(f"Auto mode query {i+1}: Bug description length: {len(query)} chars")
            result = SimilarityAPI.query(code, query)
            
            # Verify result format
            self.assertIn("score", result)
            self.assertIn("latency_sec", result)
            self.assertIsInstance(result["score"], float)
            
            print(f"  Similarity score: {result['score']:.4f}, Latency: {result['latency_sec']:.4f}s")
            results.append(result)
        
        self.assertEqual(len(results), len(self.large_code_blocks))
    
    def test_4_coderank_queries(self):
        """Test 4: Switch to CodeRankEmbed and repeat queries"""
        print("\n=== Test 4: Queries with CodeRankEmbed Model ===")
        
        # Switch to CodeRankEmbed model
        config_result = SimilarityAPI.configure(model="CodeRankEmbed")
        print(f"Switched to CodeRankEmbed: {config_result}")
        
        results = []
        for i, (code, query) in enumerate(zip(self.large_code_blocks, self.queries)):
            print(f"CodeRankEmbed query {i+1}: Bug description length: {len(query)} chars")
            result = SimilarityAPI.query(code, query)
            
            # Verify result format
            self.assertIn("score", result)
            self.assertIn("latency_sec", result)
            self.assertIsInstance(result["score"], float)
            
            print(f"  Similarity score: {result['score']:.4f}, Latency: {result['latency_sec']:.4f}s")
            results.append(result)
        
        self.assertEqual(len(results), len(self.large_code_blocks))

if __name__ == "__main__":
    unittest.main(verbosity=2)
