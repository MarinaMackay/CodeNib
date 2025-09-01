# Simple Repository

A simple test repository for testing CodeMiner traverse functionality.

## Structure

```
simple_repo/
├── main.py                    # Main entry point
├── src/                       # Source package
│   ├── __init__.py           # Package initialization
│   ├── calculator.py         # Calculator class
│   └── utils/                # Utilities package
│       ├── __init__.py       # Utils package init
│       └── helpers.py        # Helper functions
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Features

- Basic arithmetic calculator with history
- Input validation and error handling
- Modular design with utilities package
- Type hints throughout
- Comprehensive docstrings

## Usage

```python
from src.calculator import Calculator
from src.utils.helpers import format_result

calc = Calculator()
result = calc.add(10, 5)
formatted = format_result(result)
print(f"Result: {formatted}")
```

## Purpose

This repository is designed to test:
- SCIP indexing and graph generation
- Code graph traversal algorithms
- Symbol and dependency analysis
- Tree structure visualization
- Multi-hop graph exploration

The repository contains realistic Python code with:
- Import relationships
- Class definitions and methods
- Function calls and dependencies
- Package/module structure
- Type annotations
