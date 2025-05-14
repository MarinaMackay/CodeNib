# CodeMiner

## CodeMiner API
A modular FastAPI-based framework for code and query similarity services, with support for easy extension to more APIs.

### 1. Start the API Server

```python
from codeminer.api import BaseAPI

# Start the API server (registers all APIs and preloads models)
BaseAPI.start(host="127.0.0.1", port=8000, log_level="info")
```

### 2. Query Code Similarity

The `SimilarityAPI` class provides a simple interface for calculating code-to-query similarity using `jina-embeddings-v2-base-code`.

```python
from codeminer.api import SimilarityAPI

code = "def greet(name):\n    return f'Hello, {name}!'"
query = "Define a function that greets someone"

# Send a request to the running API server
result = SimilarityAPI.query(code, query)
print(result)  # Example output: {'score': 0.85, 'latency_sec': 0.05}
```

For more details, see the code and docstrings in the `codeminer/api/` directory.

