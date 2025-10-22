## Use SCIP to get the index

[SCIP](https://github.com/sourcegraph/scip/tree/main) is a code intelligence protocol for index, which has powerful support for multiple languages, e.g. python, C++, etc.
We copy the scip.proto to the local directory for convenience.

### Setup scip-python (Custom Fork)

We use a custom fork of scip-python with exclude-config support, located in `third_party/scip-python`.

#### Installation Steps

1. **Initialize the submodule** (if not already done):
   ```bash
   git submodule update --init --recursive
   ```

2. **Install dependencies and build**:
   ```bash
   cd third_party/scip-python
   npm install
   cd packages/pyright-scip
   npm install
   npm run build
   npm link
   ```

3. **Link the package globally** (so you can use `scip-python` command):
   ```bash
   npm link scip-python
   ```

#### Usage

```bash
scip-python index . --project-name=$MY_PROJECT --target-only=src/subdir
```

Related links:
- [Our scip-python fork](https://github.com/fishmingyu/scip-python/tree/exclude-config)
- [Original scip-python](https://github.com/sourcegraph/scip-python)

### Convert index.scip to index.decoded

First install the [protobuf](https://protobuf.dev/installation/).
Get the scip.proto from [SCIP](https://github.com/sourcegraph/scip/tree/main).

### Using the SCIPIndexer

The `SCIPIndexer` class provides a Python interface for working with SCIP indices. **It automatically handles conda environment isolation** to prevent conflicts with system Python packages.

#### Conda Environment Isolation

**Important:** The SCIPIndexer uses conda for environment isolation when running `scip-python`. This prevents issues with:
- Package version conflicts
- System Python package interference
- Inconsistent dependency resolution

The indexer automatically:
1. Checks if conda is installed
2. Creates a dedicated `scip-env` environment (if not exists) using [scip-environment.yml](scip-environment.yml)
3. Runs all `scip-python` commands within this isolated environment

**Manual conda environment setup** (optional - the indexer does this automatically):
```bash
conda env create -f codeminer/scip_interface/scip-environment.yml
```

#### Basic Usage

```python
from codeminer.scip_interface.scip_indexer import SCIPIndexer

# Create an indexer for a project
# By default, output goes to /tmp/<project_name>/
indexer = SCIPIndexer("/path/to/project")

# Or specify a custom output directory
indexer = SCIPIndexer("/path/to/project", output_dir="/custom/output/path")

# Generate an index (runs in isolated conda environment automatically)
indexer.generate_index(project_name="MyProject", target_dir="src")

# Decode the index.scip file to index.decoded
indexer.decode_index()

# Process the decoded index and save results
result = indexer.process_index("output.json")

# Or run the complete pipeline with one call
result = indexer.run_pipeline(
    project_name="MyProject",
    target_dir="src",
    output_file="output.json"
)
```

#### Advanced Features

**Cache Management:**
```python
# Run pipeline with cache awareness
# skip_level options: None, 'raw', 'decode', 'graph'
result = indexer.run_pipeline(
    project_name="MyProject",
    skip_level="graph"  # Reuse graph.pkl if exists
)

# Clear cache at different levels
indexer.clear_cache(level="all")     # Remove all cache files
indexer.clear_cache(level="graph")   # Keep only graph.pkl
indexer.clear_cache(level="decode")  # Keep only index.decoded
indexer.clear_cache(level="raw")     # Keep only index.scip
```

**Exclude Patterns:**
```python
# Exclude specific directories or files from indexing
indexer = SCIPIndexer(
    "/path/to/project",
    exclude_patterns=["tests/*", "*.test.py", "build/*"]
)
```
