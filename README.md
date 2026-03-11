# CodeMiner

## Setup

Install Python environment with conda:
```bash
conda create -n codeminer python=3.10
conda activate codeminer
pip install -e .
```

Optional: enable SCIP-based code indexing:
```bash
make scip
```
The setup script installs `rust-analyzer`, `scip-clang`, `@sourcegraph/scip-typescript`, and `@sourcegraph/scip-python`.

## Contribute

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

Code formatting (black, isort) and linting (flake8) will run automatically before each commit.
