"""Calculator module with basic arithmetic operations."""

from typing import Union

from .utils.helpers import validate_input


class Calculator:
    """A simple calculator class."""

    def __init__(self):
        """Initialize the calculator."""
        self.history = []

    def add(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Add two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            Sum of a and b
        """
        if not validate_input(a) or not validate_input(b):
            raise ValueError("Invalid input values")

        result = a + b
        self._record_operation("add", a, b, result)
        return result

    def subtract(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Subtract b from a.

        Args:
            a: First number
            b: Second number

        Returns:
            Difference of a and b
        """
        if not validate_input(a) or not validate_input(b):
            raise ValueError("Invalid input values")

        result = a - b
        self._record_operation("subtract", a, b, result)
        return result

    def multiply(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Multiply two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            Product of a and b
        """
        if not validate_input(a) or not validate_input(b):
            raise ValueError("Invalid input values")

        result = a * b
        self._record_operation("multiply", a, b, result)
        return result

    def divide(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Divide a by b.

        Args:
            a: Dividend
            b: Divisor

        Returns:
            Quotient of a and b

        Raises:
            ValueError: If b is zero or inputs are invalid
        """
        if not validate_input(a) or not validate_input(b):
            raise ValueError("Invalid input values")

        if b == 0:
            raise ValueError("Cannot divide by zero")

        result = a / b
        self._record_operation("divide", a, b, result)
        return result

    def _record_operation(
        self,
        operation: str,
        a: Union[int, float],
        b: Union[int, float],
        result: Union[int, float],
    ):
        """Record an operation in history.

        Args:
            operation: Name of the operation
            a: First operand
            b: Second operand
            result: Result of the operation
        """
        self.history.append(
            {"operation": operation, "operands": [a, b], "result": result}
        )

    def get_history(self) -> list:
        """Get calculation history.

        Returns:
            List of previous calculations
        """
        return self.history.copy()

    def clear_history(self):
        """Clear calculation history."""
        self.history.clear()
