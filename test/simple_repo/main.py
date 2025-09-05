"""Main module for the simple test repository."""

from src.calculator import Calculator
from src.utils.helpers import format_result, validate_input


def main():
    """Main function that demonstrates the calculator."""
    calc = Calculator()

    # Validate inputs
    a = 10
    b = 5

    if not validate_input(a) or not validate_input(b):
        print("Invalid input!")
        return

    # Perform calculations
    result = calc.add(a, b)
    formatted = format_result(result)
    print(f"Addition: {formatted}")

    result = calc.multiply(a, b)
    formatted = format_result(result)
    print(f"Multiplication: {formatted}")

    # Division with error handling
    try:
        result = calc.divide(a, b)
        formatted = format_result(result)
        print(f"Division: {formatted}")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
