# CodeMiner

## Setup

Install Python environment with conda:
```bash
conda env create -f pyproject.toml
conda activate codeminer
```

Install scip:
```bash
npm install -g @sourcegraph/scip-python
```

## Examples

Baseline: Search + Rerank Pipeline

```python
repo_path = ''
pipeline = SearchRerankPipeline(
    repo_path=repo_path,
    embedding_model="nomic-ai/CodeRankEmbed",
    embedding_provider="huggingface",
    embedding_dimension=768,
    rerank_model="nomic-ai/CodeRankLLM",
    languages=['python'],
    max_lines_per_chunk=100,
)

results = pipeline.query(query=problem_statement, top_k=5)
```
