"""Helper functions for the calculator."""

import functools
import time
from typing import Union


def timer(func):
    """Decorator to measure function execution time."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result

    return wrapper


def validate_positive(func):
    """Decorator to ensure arguments are positive numbers."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            if isinstance(arg, (int, float)) and arg < 0:
                raise ValueError(f"Arguments must be positive, got {arg}")
        return func(*args, **kwargs)

    return wrapper


def validate_input(value: Union[int, float]) -> bool:
    """Validate if input is a valid number.

    Args:
        value: Value to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(value, (int, float)):
        return False

    # Check for NaN and infinity
    if isinstance(value, float):
        import math

        if math.isnan(value) or math.isinf(value):
            return False

    return True


def format_result(result: Union[int, float]) -> str:
    """Format calculation result for display.

    Args:
        result: Calculation result

    Returns:
        Formatted string representation
    """

    def _remove_trailing_zeros(value_str: str) -> str:
        """Helper function to remove trailing zeros from decimal."""
        return value_str.rstrip("0").rstrip(".")

    if isinstance(result, float):
        # Round to 4 decimal places and remove trailing zeros
        formatted = f"{result:.4f}"
        return _remove_trailing_zeros(formatted)
    else:
        return str(result)


def is_even(number: int) -> bool:
    """Check if a number is even.

    Args:
        number: Integer to check

    Returns:
        True if even, False if odd
    """
    return number % 2 == 0


@timer
@validate_positive
def factorial(n: int) -> int:
    """Calculate factorial of a number.

    Args:
        n: Non-negative integer

    Returns:
        Factorial of n

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")

    if n == 0 or n == 1:
        return 1

    result = 1
    for i in range(2, n + 1):
        result *= i

    return result


async def async_calculate(x: int, y: int) -> int:
    """Async function to calculate sum.

    Args:
        x: First number
        y: Second number

    Returns:
        Sum of x and y
    """
    import asyncio

    await asyncio.sleep(0.1)
    return x + y


@timer
async def async_factorial(n: int) -> int:
    """Async decorated factorial calculation.

    Args:
        n: Non-negative integer

    Returns:
        Factorial of n
    """
    import asyncio

    await asyncio.sleep(0.1)
    return factorial(n)
