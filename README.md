# CodeMiner

## CodeMiner API
A modular FastAPI-based framework for code and query similarity services, with support for easy extension to more APIs.

### 1. Query Code Similarity

The `SimilarityAPI` class provides a simple interface for calculating code-to-query similarity using the following embedding models:
- jina-embeddings-v2-base-code (default)
- CodeRankEmbed

```python
from codeminer.api import SimilarityAPI

code = "def greet(name):\n    return f'Hello, {name}!'"
query = "Define a function that greets someone"

# Send a request to the running API server
result = SimilarityAPI.query(code, query)
print(result)  # Example output: {'score': 0.85, 'latency_sec': 0.05}
```
### 2. Model Switching

You can switch between different models and devices at runtime:

```python
# Switch to CUDA device
SimilarityAPI.configure(device="cuda")

# Switch to CodeRankEmbed model
SimilarityAPI.configure(model="CodeRankEmbed")

# Switch both model and device simultaneously
SimilarityAPI.configure(model="CodeRankEmbed", device="cuda")

```




