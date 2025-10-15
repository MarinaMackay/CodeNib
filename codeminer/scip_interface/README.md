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

The `SCIPIndexer` class provides a Python interface for working with SCIP indices.

```python
from codeminer.scip_interface.scip_indexer import SCIPIndexer

# Create an indexer for a project
indexer = SCIPIndexer("/path/to/project")

# Generate an index (returns True if successful)
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
