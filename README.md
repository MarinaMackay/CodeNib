# CodeMiner

## SimilarityAPI

The `SimilarityAPI` class provides a simple interface for calculating code-to-query similarity using `jina-embeddings-v2-base-code`.

### API Reference

#### SimilarityAPI.start()

Starts the similarity API server in a background thread.

The server will automatically shut down when your program exits.

```python
SimilarityAPI.start(
    host="127.0.0.1",     # API server host (default: 127.0.0.1)
    port=8000,            # API server port (default: 8000)
    log_level="warning"   # Log level (default: warning)
)
```

#### SimilarityAPI.query()

Calculates the similarity between a code snippet and a natural language query.

```python
result = SimilarityAPI.query(
    code="def add(a, b): return a + b",
    query="function to add two numbers"
)
```

Return a dictionary containing:
- `score` (float): Similarity score between 0 and 1, where higher values indicate greater similarity
- `latency_sec` (float): Time taken to calculate the similarity, in seconds
