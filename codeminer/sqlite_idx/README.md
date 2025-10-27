# SQLite Node Index

Lightweight in-memory SQLite index for fast querying of CodeGraph nodes.

- **In-memory database**: Fast queries, supports arbitrary SQL statements
- **GLOB matching**: Supports pattern matching (e.g., `*calculator*`)

## Quick Start

```python
from codeminer import CodeGraph, SqliteNodeIndex

# Build index from existing CodeGraph
code_graph = CodeGraph.load_graph("~/.codeminer/xxx/graph.pkl")
idx = SqliteNodeIndex(code_graph=code_graph)

# SQL query examples
# 1. Find all file nodes
files = idx.query("SELECT node_name FROM nodes WHERE type='file'")

# 2. GLOB pattern matching
calc_nodes = idx.query(
    "SELECT node_name, type FROM nodes WHERE node_name GLOB '*calculator*'"
)

# 3. Complex queries
stats = idx.query(
    "SELECT type, COUNT(*) as cnt FROM nodes GROUP BY type ORDER BY cnt DESC"
)

# 4. Query statistics
stats = idx.get_stats()
print(f"Total nodes: {stats['total_nodes']}")
print(f"Nodes by type: {stats['nodes_by_type']}")
```

## Database Schema

### nodes table

| Column | Type | Description |
|--------|------|-------------|
| vertex_id | INTEGER | Primary key, corresponds to igraph node ID |
| node_name | TEXT | Node name (e.g., `src/calculator.py:Calculator.add()`) |
| type | TEXT | Node type (file/class/function/method/field/directory/root) |
| file | TEXT | File path |
| start_line | INTEGER | Start line number |
| end_line | INTEGER | End line number |
| content | TEXT | Source code content (truncated to 8000 characters) |

### Indexes

- `idx_node_name`: Accelerates queries by name

## TODO

- [ ] Sqlite should be initialized by the schema and could add more than one `CodeGraph` instance later. (But how to distinguish from different repos? `instance_id`)
- [ ] Add disk persistence support (save/load functionality)
- [ ] Index settings (idx_node_type, idx_node_file)


