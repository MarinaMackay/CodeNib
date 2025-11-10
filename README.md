# CodeMiner

## Setup

Install Python environment:
```bash
conda create -n codeminer python=3.10
conda activate codeminer
pip install -e .
```

Install scip for code indexing:
```bash
npm install -g @sourcegraph/scip-python
```

## Contribute

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

Code formatting (black, isort) and linting (flake8) will run automatically before each commit.
