## Use SCIP to get the index

[SCIP](https://github.com/sourcegraph/scip/tree/main) is a code intelligence protocol for index, which has powerful support for multiple languages, e.g. python, C++, etc.
We copy the scip.proto to the local directory for convenience.

Related links are listed as below:
[scip-python](https://github.com/sourcegraph/scip-python)
Usage: 
``` bash
scip-python index . --project-name=$MY_PROJECT --target-only=src/subdir
```

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

### Command Line Usage

The indexer can also be used from the command line:

```bash
python -m codeminer.scip_interface.scip_indexer --project-dir /path/to/project --project-name "MyProject"
```

Options:
- `--project-dir` - Path to the project root directory
- `--project-name` - Project name for the index
- `--target-dir` - Subdirectory to target for indexing
- `--output` - Path to output processed index file
- `--skip-index` - Skip index generation, use existing index.scip
- `--skip-decode` - Skip decoding, use existing index.decoded