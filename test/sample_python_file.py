#!/usr/bin/env python3
"""
Test Python file for code chunking functionality.
This file contains various Python constructs to test the chunker.
"""

# Global variable
GLOBAL_CONFIG = {"debug": True, "version": "1.0.0"}


def hello_world():
    """Simple function to say hello."""
    return "Hello, World!"


class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.result = 0

    def add(self, x, y):
        """Add two numbers."""
        self.result = x + y
        return self.result

    def subtract(self, x, y):
        """Subtract y from x."""
        self.result = x - y
        return self.result


def fibonacci(n):
    """Calculate fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


class DataProcessor:
    """Process data with various methods."""

    def __init__(self, data=None):
        self.data = data or []

    def process(self):
        """Process the data."""
        return [x * 2 for x in self.data]


async def async_function():
    """An async function example."""
    import asyncio

    await asyncio.sleep(1)
    return "Done"


# Some code at the end
if __name__ == "__main__":
    calc = Calculator()
    result = calc.add(5, 3)
    print(f"Result: {result}")
