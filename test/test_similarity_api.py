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
            {"code": """
                def subtract_with_validation(a, b, allow_negative=True):
                    '''
                    Subtracts b from a with optional validation.
                    
                    Args:
                        a: First operand
                        b: Second operand
                        allow_negative: If False, returns 0 when result would be negative
                    
                    Returns:
                        The result of a - b, or 0 if result is negative and allow_negative is False
                    '''
                    result = a - b
                    
                    # Apply validation logic if needed
                    if not allow_negative and result < 0:
                        return 0
                    
                    # Log the operation for debugging purposes
                    print(f"Performed subtraction: {a} - {b} = {result}")
                    
                    return result
                """, 
             "query": "A mathematical function that performs subtraction of two numbers with an optional parameter to prevent negative results, includes input validation and operation logging for debugging"},

            {"code": """
                def divide_with_error_handling(a, b, precision=2):
                    '''
                    Safely divides a by b with error handling and precision control.
                    
                    Args:
                        a: Numerator
                        b: Denominator
                        precision: Number of decimal places to round to
                    
                    Returns:
                        The result of a / b rounded to specified precision,
                        or appropriate error messages for invalid inputs
                    
                    Raises:
                        ZeroDivisionError: If b is zero
                        TypeError: If inputs are not numeric
                    '''
                    # Type checking
                    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                        raise TypeError("Both inputs must be numeric")
                    
                    # Handle division by zero
                    if b == 0:
                        raise ZeroDivisionError("Cannot divide by zero")
                    
                    # Perform division and round to specified precision
                    result = a / b
                    rounded_result = round(result, precision)
                    
                    return rounded_result
                """, 
             "query": "A robust division function that handles potential errors such as division by zero and non-numeric inputs, while allowing control over the decimal precision of the result through rounding"},

            {"code": """
                def calculate_statistics(numbers):
                    '''
                    Calculates various statistics for a collection of numbers.
                    
                    Args:
                        numbers: A list or tuple of numeric values
                    
                    Returns:
                        A dictionary containing calculated statistics including:
                        - count: Number of elements
                        - sum: Sum of all values
                        - mean: Average value
                        - median: Middle value (or average of two middle values)
                        - min: Minimum value
                        - max: Maximum value
                        - range: Difference between max and min
                    '''
                    # Validate input
                    if not numbers:
                        return {
                            "count": 0,
                            "sum": 0,
                            "mean": None,
                            "median": None,
                            "min": None,
                            "max": None,
                            "range": None
                        }
                    
                    # Sort the list for median calculation
                    sorted_nums = sorted(numbers)
                    count = len(sorted_nums)
                    
                    # Calculate basic statistics
                    total = sum(sorted_nums)
                    mean = total / count
                    min_val = sorted_nums[0]
                    max_val = sorted_nums[-1]
                    
                    # Calculate median
                    if count % 2 == 0:
                        # Even number of elements
                        middle_right = count // 2
                        middle_left = middle_right - 1
                        median = (sorted_nums[middle_left] + sorted_nums[middle_right]) / 2
                    else:
                        # Odd number of elements
                        median = sorted_nums[count // 2]
                    
                    return {
                        "count": count,
                        "sum": total,
                        "mean": mean,
                        "median": median,
                        "min": min_val,
                        "max": max_val,
                        "range": max_val - min_val
                    }
                """, 
             "query": "A comprehensive statistical analysis function that takes a list of numbers and computes multiple statistics including count, sum, mean, median, minimum, maximum, and range, with proper handling of edge cases like empty input"},

            {"code": """
                def validate_password(password, min_length=8, require_special=True, 
                                    require_uppercase=True, require_numbers=True):
                    '''
                    Validates a password against security requirements.
                    
                    Args:
                        password: The password string to validate
                        min_length: Minimum required length
                        require_special: Whether to require special characters
                        require_uppercase: Whether to require uppercase letters
                        require_numbers: Whether to require numeric digits
                    
                    Returns:
                        A tuple containing:
                        - Boolean indicating if password is valid
                        - List of validation errors (empty if valid)
                    '''
                    errors = []
                    
                    # Check password length
                    if len(password) < min_length:
                        errors.append(f"Password must be at least {min_length} characters long")
                    
                    # Check for uppercase letters if required
                    if require_uppercase and not any(char.isupper() for char in password):
                        errors.append("Password must contain at least one uppercase letter")
                    
                    # Check for numbers if required
                    if require_numbers and not any(char.isdigit() for char in password):
                        errors.append("Password must contain at least one number")
                    
                    # Check for special characters if required
                    if require_special:
                        special_chars = "!@#$%^&*()-_=+[]{}|;:,.<>?/~`"
                        if not any(char in special_chars for char in password):
                            errors.append("Password must contain at least one special character")
                    
                    # Return validation result
                    is_valid = len(errors) == 0
                    return (is_valid, errors)
                """, 
             "query": "A security utility function that evaluates password strength based on customizable criteria such as minimum length, presence of uppercase letters, numbers, and special characters, returning both validation status and specific failure reasons"},

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
