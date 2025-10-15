# CodeMiner Examples

## (baseline) Search + Rerank (S+R) pipeline

The Search + Rerank (S+R) pipeline is a two-stage retrieval system:
1. **Search**: Uses embedding-based vector search to retrieve top-K candidate code chunks
2. **Rerank**: Uses an LLM to rerank candidates based on relevance to the query

Before running the pipeline, start a vLLM server for the rerank model:

```bash
python scripts/start_vllm_server.py --model Qwen/Qwen2.5-Coder-7B
```

```python
from examples import SearchRerankPipeline

# Initialize the pipeline
pipeline = SearchRerankPipeline(
    repo_path="/path/to/your/repository",
    repo_commit="commit_hash",
    embedding_model="nomic-ai/CodeRankEmbed",
    embedding_provider="huggingface",
    embedding_dimension=768,
    rerank_model="Qwen/Qwen2.5-Coder-7B",
    languages=["python"],
    max_lines_per_chunk=100,
)

# Query the pipeline
query = "How do I fix the bug in data processing?"
results = pipeline.query(query=query, top_k=5)

# Access results
for i, node in enumerate(results):
    print(f"Rank {i+1} (Score: {node.score:.4f})")
    print(f"  File: {node.file}")
    print(f"  Lines: {node.start_line}-{node.end_line}")
    print(f"  Node: {node.node_name}")
```

