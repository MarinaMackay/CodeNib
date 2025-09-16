"""Helper functions for the calculator."""

from typing import Union


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
