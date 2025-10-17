import argparse
from pathlib import Path

from codeminer.env import load_filter_locbench_dataset, process_locbench_instance
from codeminer.graph.transverse_graph import traverse_tree_structure
from codeminer.scip_interface import SCIPIndexer

args_dict = {
    "model": "gpt-4o",
    "dataset": "czlll/Loc-Bench_V1",
    "split": "test",
    "filter_instance": "^(sympy__sympy-27223)$",
}


def test_scip_exclude():
    exclude_file = "sympy/polys/numberfields/resolvent_lookup.py"
    exclude_pattern = "test_*"
    args = argparse.Namespace(**args_dict)
    dataset = load_filter_locbench_dataset(args=args)

    instance = dataset[0]
    repo_path = process_locbench_instance(instance)
    # set output path with ~/.codeminer/instance_id
    output_path = str(Path.home()) + "/.codeminer/" + instance["instance_id"]
    # setup codegraph with exclude patterns
    repo_indexer = SCIPIndexer(
        repo_path,
        output_dir=output_path,
        exclude_patterns=[exclude_file, exclude_pattern],
    )

    # Run the indexing pipeline from scratch (skip_level=None)
    graph = repo_indexer.run_pipeline(project_name="test_swebench", skip_level="graph")

    # list the neighbors of the sympy/utilities/lamdify.py file node
    start_file = "sympy/utilities/lambdify.py"

    # Test downstream traversal
    tree_result = traverse_tree_structure(
        graph,
        start_file,
        direction="downstream",
        hops=1,
    )

    print(f"Tree structure (downstream, 1 hop):")
    print(tree_result)

    assert tree_result is not None
    assert isinstance(tree_result, (dict, list, str))
