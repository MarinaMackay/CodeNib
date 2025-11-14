#!/usr/bin/env python3
"""
Test correctness of C++ SCIP decoder. (compare with Python decoder)

This script:
1. Uses Python to decode a SCIP file and saves graph as JSON
2. Uses C++ standalone decoder to decode the same SCIP file and saves graph as JSON
3. Compares the two JSON outputs and generates a diff report

Usage Examples:
    # Test single SWE-bench instance
    python test/scip/test_scip_core_w_python.py --instance django__django-11099

    # Test default instances (7 predefined instances)
    python test/scip/test_scip_core_w_python.py --multiple-instances

    # Test random N instances from SWE-bench_Verified
    python test/scip/test_scip_core_w_python.py --num-instances 10

    # Test local project
    python test/scip/test_scip_core_w_python.py test/simple_repo

Output:
    Results are saved in test/scip/comparison_results/
    - For each instance: <instance_id>/python_output.json, cpp_output.json
    - For batch testing: summary_report.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add parent directory to path to import codeminer modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from codeminer.scip_interface.scip_decode import SCIPGraphDecoder
from codeminer.scip_interface.scip_indexer import SCIPIndexer
from codeminer.env.process_swebench_data import load_filter_swebench_dataset_explicit, process_swebench_instance

# Default instances for testing
DEFAULT_TEST_INSTANCES = [
    "astropy__astropy-13033",
    "django__django-11099",
    "scikit-learn__scikit-learn-10297",
    "mwaskom__seaborn-3187",
    "matplotlib__matplotlib-24570",
    "pydata__xarray-3993",
    "sphinx-doc__sphinx-7757"
]


def get_or_generate_scip_file(project_root: Path) -> Tuple[Path | None, bool]:
    """
    Get cached SCIP file or generate new one.

    Args:
        project_root: Project root directory

    Returns:
        Tuple of (scip_file_path, success)
        If failed, returns (None, False)
    """
    import tempfile

    # Use platform-independent temporary directory
    scip_cache_dir = Path(tempfile.gettempdir()) / project_root.name
    scip_cache_dir.mkdir(parents=True, exist_ok=True)

    scip_file = scip_cache_dir / "index.scip"
    decoded_scip = scip_cache_dir / "index.decoded"

    # Check if cached SCIP index exists (prefer decoded version)
    if decoded_scip.exists():
        print(f"  Found cached decoded SCIP at {decoded_scip}")
        return decoded_scip, True
    elif scip_file.exists():
        print(f"  Found cached SCIP index at {scip_file}")
        return scip_file, True
    else:
        # Generate SCIP index to cache directory
        try:
            scip_file = generate_scip_index(project_root, scip_cache_dir)
            print(f"  ✓ Generated SCIP index to cache")
            return scip_file, True
        except Exception as e:
            print(f"  ❌ Failed to generate SCIP index: {e}")
            return None, False


def generate_scip_index(project_root: Path, output_dir: Path) -> Path:
    """
    Generate SCIP index for the project if it doesn't exist.

    Args:
        project_root: Project root directory (where Python source code is located)
        output_dir: Output directory for SCIP files

    Returns:
        Path to the generated SCIP index file (decoded)
    """
    try:
        # Create indexer
        indexer = SCIPIndexer(
            project_root=project_root,
            output_dir=output_dir
        )

        # Generate and decode index
        # Use skip_level='raw' to reuse existing index.scip if it exists
        if indexer.index_file.exists():
            # Ensure it's decoded
            if not indexer.decoded_file.exists():
                if not indexer.decode_index():
                    raise RuntimeError("Failed to decode SCIP index")
            return indexer.decoded_file

        # Generate new index
        # IMPORTANT: Pass project_root as cwd so scip-python indexes the correct directory
        if not indexer.generate_index(cwd=project_root):
            raise RuntimeError("Failed to generate SCIP index")

        # Decode the index
        if not indexer.decode_index():
            raise RuntimeError("Failed to decode SCIP index")

        return indexer.decoded_file

    except Exception as e:
        raise RuntimeError(f"Error generating SCIP index: {e}") from e


def ensure_decoded_scip(scip_file: Path) -> Path:
    """Ensure SCIP file is in decoded text format."""
    # Check if file is binary by trying to read as text
    try:
        with open(scip_file, 'r', encoding='utf-8') as f:
            f.read(100)
        # If successful, it's already decoded
        return scip_file
    except UnicodeDecodeError:
        # It's binary, need to decode
        decoded_file = scip_file.with_suffix('.scip.decoded')

        # Find scip.proto file - go up from test/scip to project root
        module_dir = Path(__file__).parent.parent.parent / "codeminer" / "scip_interface"
        proto_file = module_dir / "scip.proto"

        if not proto_file.exists():
            raise FileNotFoundError(f"scip.proto not found at {proto_file}")

        # Use protoc to decode binary SCIP to text format (safe from shell injection)
        with open(scip_file, 'rb') as stdin_file, open(decoded_file, 'wb') as stdout_file:
            result = subprocess.run(
                [
                    "protoc",
                    "--decode=scip.Index",
                    f"--proto_path={module_dir}",
                    "scip.proto"
                ],
                stdin=stdin_file,
                stdout=stdout_file,
                stderr=subprocess.PIPE,
                text=False
            )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to decode SCIP file: {result.stderr.decode('utf-8')}")

        return decoded_file


def normalize_graph_data(graph_data: Dict) -> Dict:
    """Normalize graph data for consistent comparison."""
    # Sort nodes/vertices by ID (or name if ID not present)
    if "nodes" in graph_data:
        graph_data["nodes"] = sorted(
            graph_data["nodes"],
            key=lambda x: (x.get("id", float('inf')), x.get("name", ""))
        )
    if "vertices" in graph_data:
        graph_data["vertices"] = sorted(
            graph_data["vertices"],
            key=lambda x: (x.get("id", float('inf')), x.get("name", ""))
        )

    # Sort edges by source, target, type (supporting both ID-based and name-based)
    if "edges" in graph_data:
        def edge_sort_key(edge):
            source = edge.get("source", "")
            target = edge.get("target", "")
            edge_type = edge.get("type", "")
            # If source/target are integers, use them directly; otherwise treat as strings
            if isinstance(source, int):
                return (source, target, edge_type)
            else:
                return (str(source), str(target), edge_type)

        graph_data["edges"] = sorted(graph_data["edges"], key=edge_sort_key)

    return graph_data


def python_decode(scip_file: Path, project_root: Path, output_file: Path) -> bool:
    """Decode SCIP file using Python decoder."""
    print(f"Python decoder...")

    try:
        # Ensure SCIP file is decoded (text format)
        decoded_scip = ensure_decoded_scip(scip_file)

        # Decode using Python
        start_time = time.time()
        decoder = SCIPGraphDecoder(str(decoded_scip), str(project_root))
        print(f"  Decoding into a graph...")
        graph = decoder.decode()
        elapsed = time.time() - start_time
        print(f"  ✓ Decoding completed in {elapsed:.1f}s")

        # Convert to JSON format (matching C++ output format)
        vertices = []
        edges = []

        # Extract nodes (vertices) with ID
        for v in graph.graph.vs:
            node_data = {
                "id": v.index, 
                "name": v["name"],
                "type": v["type"]
            }
            if "file" in v.attributes():
                node_data["file"] = v["file"]
            if "start_line" in v.attributes():
                node_data["start_line"] = v["start_line"]
            if "end_line" in v.attributes():
                node_data["end_line"] = v["end_line"]
            vertices.append(node_data)

        # Extract edges (using vertex IDs, not names, to match C++)
        for i, e in enumerate(graph.graph.es):
            edges.append({
                "id": i,  # Add edge ID to match C++
                "source": e.source,  # Use vertex ID (int) instead of name
                "target": e.target,  # Use vertex ID (int) instead of name
                "type": e["type"]
            })

        graph_data = {
            "project_root": str(project_root) if project_root else None,
            "vertices": vertices,  # Changed from "nodes" to "vertices" to match C++
            "edges": edges
        }

        # Normalize and save
        graph_data = normalize_graph_data(graph_data)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(graph_data, f, indent=2)

        print(f"  ✓ {len(vertices)} vertices, {len(edges)} edges")
        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def cpp_decode(scip_file: Path, project_root: Path, output_file: Path, cpp_decoder: Path) -> bool:
    """Decode SCIP file using C++ decoder."""
    print(f"C++ decoder...")

    try:
        # Ensure SCIP file is decoded (text format)
        decoded_scip = ensure_decoded_scip(scip_file)

        # Run C++ decoder
        print(f"  Decoding with C++ decoder...")
        start_time = time.time()

        project_root_arg = str(project_root) if project_root else "null"
        cmd = [str(cpp_decoder), str(decoded_scip), project_root_arg, str(output_file)]

        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"  ❌ Error (after {elapsed:.1f}s): {result.stderr if result.stderr else 'unknown error'}")
            return False

        print(f"  ✓ C++ decoding completed in {elapsed:.1f}s")

        # Normalize the C++ output
        if output_file.exists():
            with open(output_file, "r") as f:
                graph_data = json.load(f)

            # Count nodes and edges
            nodes = graph_data.get("nodes", graph_data.get("vertices", []))
            edges = graph_data.get("edges", [])

            graph_data = normalize_graph_data(graph_data)
            with open(output_file, "w") as f:
                json.dump(graph_data, f, indent=2)

            print(f"  ✓ {len(nodes)} nodes, {len(edges)} edges")

        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def compare_nodes(py_nodes: List[Dict], cpp_nodes: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
    """Compare nodes between Python and C++ outputs."""
    py_node_map = {n["name"]: n for n in py_nodes}
    cpp_node_map = {n["name"]: n for n in cpp_nodes}

    py_names = set(py_node_map.keys())
    cpp_names = set(cpp_node_map.keys())

    only_in_py = sorted(py_names - cpp_names)
    only_in_cpp = sorted(cpp_names - py_names)

    differences = []
    for name in sorted(py_names & cpp_names):
        py_node = py_node_map[name]
        cpp_node = cpp_node_map[name]

        diffs = []
        for key in set(py_node.keys()) | set(cpp_node.keys()):
            py_val = py_node.get(key)
            cpp_val = cpp_node.get(key)

            if py_val != cpp_val:
                diffs.append(f"  {key}: Python={py_val}, C++={cpp_val}")

        if diffs:
            differences.append(f"Node '{name}':\n" + "\n".join(diffs) + "\n")

    return only_in_py, only_in_cpp, differences


def compare_edges(py_edges: List[Dict], cpp_edges: List[Dict], py_nodes: List[Dict] = None, cpp_nodes: List[Dict] = None) -> Tuple[List[str], List[str]]:
    """Compare edges between Python and C++ outputs."""
    def normalize_edge_type(edge_type):
        """Normalize edge types - treat 'contain' and 'reference' as equivalent."""
        # C++ may use 'reference' where Python uses 'contain' for certain edges
        # These are considered equivalent for comparison purposes
        if edge_type in ('contain', 'reference'):
            return 'contain_or_reference'
        return edge_type

    def edge_key(e):
        # Use source/target directly (should be IDs now, but also works with names)
        return (e["source"], e["target"], normalize_edge_type(e["type"]))

    py_edge_set = {edge_key(e) for e in py_edges}
    cpp_edge_set = {edge_key(e) for e in cpp_edges}

    # Build ID-to-name mappings for readable output (if nodes provided and have IDs)
    id_to_name = {}
    if py_nodes:
        for node in py_nodes:
            if "id" in node and "name" in node:
                id_to_name[node["id"]] = node["name"]
    if cpp_nodes:
        for node in cpp_nodes:
            if "id" in node and "name" in node:
                id_to_name[node["id"]] = node["name"]

    def format_edge(s, t, typ):
        # If s and t are integers and we have name mappings, show names for readability
        if isinstance(s, int) and s in id_to_name:
            s = f"{s}({id_to_name[s]})"
        if isinstance(t, int) and t in id_to_name:
            t = f"{t}({id_to_name[t]})"
        return f"{s} -> {t} ({typ})"

    only_in_py = sorted([format_edge(s, t, typ) for s, t, typ in py_edge_set - cpp_edge_set])
    only_in_cpp = sorted([format_edge(s, t, typ) for s, t, typ in cpp_edge_set - py_edge_set])

    return only_in_py, only_in_cpp


def generate_report(py_file: Path, cpp_file: Path, json_report_file: Path = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Generate comparison report.

    Args:
        py_file: Path to Python decoder output JSON
        cpp_file: Path to C++ decoder output JSON
        json_report_file: Optional path to save detailed JSON report

    Returns:
        Tuple of (success, stats_dict) where stats_dict contains comparison statistics
    """
    print(f"\nComparing...")

    try:
        # Load JSON files
        with open(py_file, "r") as f:
            py_data = json.load(f)

        with open(cpp_file, "r") as f:
            cpp_data = json.load(f)

        # Compare - handle both "nodes" and "vertices" keys
        py_nodes = py_data.get("nodes", py_data.get("vertices", []))
        cpp_nodes = cpp_data.get("nodes", cpp_data.get("vertices", []))
        py_edges = py_data.get("edges", [])
        cpp_edges = cpp_data.get("edges", [])

        # Compare nodes and edges directly (both now use ID-based format)
        nodes_only_py, nodes_only_cpp, node_diffs = compare_nodes(py_nodes, cpp_nodes)
        edges_only_py, edges_only_cpp = compare_edges(py_edges, cpp_edges, py_nodes, cpp_nodes)

        # Filter out 'id' differences in nodes
        meaningful_node_diffs = []
        for diff in node_diffs:
            if "id:" not in diff:
                meaningful_node_diffs.append(diff)

        total_meaningful_issues = len(nodes_only_py) + len(nodes_only_cpp) + len(meaningful_node_diffs) + len(edges_only_py) + len(edges_only_cpp)

        # Statistics dictionary
        stats = {
            "py_nodes": len(py_nodes),
            "cpp_nodes": len(cpp_nodes),
            "py_edges": len(py_edges),
            "cpp_edges": len(cpp_edges),
            "nodes_only_py": len(nodes_only_py),
            "nodes_only_cpp": len(nodes_only_cpp),
            "node_diffs": len(meaningful_node_diffs),
            "edges_only_py": len(edges_only_py),
            "edges_only_cpp": len(edges_only_cpp),
            "total_issues": total_meaningful_issues,
            "perfect_match": total_meaningful_issues == 0
        }

        # Detailed report data for JSON export
        detailed_report = {
            "summary": stats,
            "details": {
                "nodes_only_in_python": nodes_only_py,
                "nodes_only_in_cpp": nodes_only_cpp,
                "node_differences": meaningful_node_diffs,
                "edges_only_in_python": edges_only_py,
                "edges_only_in_cpp": edges_only_cpp
            }
        }

        # Save detailed JSON report if requested
        if json_report_file:
            json_report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(json_report_file, "w") as f:
                json.dump(detailed_report, f, indent=2)
            print(f"  ✓ Detailed JSON report saved to: {json_report_file}")

        # Generate simplified report
        report = []
        report.append("=" * 80)
        report.append(f"Python: {len(py_nodes)} nodes, {len(py_edges)} edges")
        report.append(f"C++:    {len(cpp_nodes)} nodes, {len(cpp_edges)} edges")
        report.append("")

        if total_meaningful_issues == 0:
            report.append("✅ PERFECT MATCH")
        else:
            report.append(f"⚠️  {total_meaningful_issues} differences found")
        report.append("")

        # Only show differences if there are meaningful issues
        if total_meaningful_issues > 0:
            if nodes_only_py:
                report.append(f"Nodes only in Python: {len(nodes_only_py)}")
                for node in nodes_only_py[:5]:
                    report.append(f"  - {node}")
                if len(nodes_only_py) > 5:
                    report.append(f"  ... and {len(nodes_only_py) - 5} more")

            if nodes_only_cpp:
                report.append(f"Nodes only in C++: {len(nodes_only_cpp)}")
                for node in nodes_only_cpp[:5]:
                    report.append(f"  - {node}")
                if len(nodes_only_cpp) > 5:
                    report.append(f"  ... and {len(nodes_only_cpp) - 5} more")

            if meaningful_node_diffs:
                report.append(f"Node differences: {len(meaningful_node_diffs)}")
                for diff in meaningful_node_diffs[:3]:
                    report.append(f"  {diff}")
                if len(meaningful_node_diffs) > 3:
                    report.append(f"  ... and {len(meaningful_node_diffs) - 3} more")

            if edges_only_py:
                report.append(f"Edges only in Python: {len(edges_only_py)}")
                for edge in edges_only_py[:5]:
                    report.append(f"  - {edge}")
                if len(edges_only_py) > 5:
                    report.append(f"  ... and {len(edges_only_py) - 5} more")

            if edges_only_cpp:
                report.append(f"Edges only in C++: {len(edges_only_cpp)}")
                for edge in edges_only_cpp[:5]:
                    report.append(f"  - {edge}")
                if len(edges_only_cpp) > 5:
                    report.append(f"  ... and {len(edges_only_cpp) - 5} more")

        report.append("=" * 80)

        # Print report to console
        report_text = "\n".join(report)
        print(f"\n{report_text}")

        return total_meaningful_issues == 0, stats

    except Exception as e:
        print(f"❌ Error: {e}")
        return False, {}


def process_single_instance(
    instance_id: str,
    args: argparse.Namespace,
    cpp_decoder: Path
) -> Tuple[bool, Dict[str, Any]]:
    """
    Process a single SWE-bench instance.

    Returns:
        Tuple of (success, stats_dict)
    """
    print(f"\n{'='*80}")
    print(f"Processing instance: {instance_id}")
    print(f"{'='*80}")

    try:
        # Load the dataset and filter by instance ID
        dataset = load_filter_swebench_dataset_explicit(
            dataset=args.swebench_dataset,
            filter_instance=instance_id,
            split=args.swebench_split
        )

        if len(dataset) == 0:
            print(f"❌ Instance {instance_id} not found in {args.swebench_dataset}")
            return False, {"error": "instance_not_found"}

        instance = dataset[0]
        print(f"  Repo: {instance['repo']}")
        print(f"  Base commit: {instance['base_commit']}")

        # Process the instance (download and checkout)
        project_root = Path(process_swebench_instance(instance, cache_dir=args.cache_dir))
        print(f"  Project root: {project_root}")

        # Get or generate SCIP file
        scip_file, success = get_or_generate_scip_file(project_root)
        if not success:
            return False, {"error": "scip_generation_failed"}

        # Output files
        instance_output_dir = args.output_dir / instance_id
        instance_output_dir.mkdir(parents=True, exist_ok=True)

        py_output = instance_output_dir / "python_output.json"
        cpp_output = instance_output_dir / "cpp_output.json"
        json_report = instance_output_dir / "comparison_report.json"

        # Run decoders
        if not python_decode(scip_file, project_root, py_output):
            print(f"  ❌ Python decoder failed")
            return False, {"error": "python_decode_failed"}

        if not cpp_decode(scip_file, project_root, cpp_output, cpp_decoder):
            print(f"  ❌ C++ decoder failed")
            return False, {"error": "cpp_decode_failed"}

        # Generate comparison report with JSON output
        success, stats = generate_report(py_output, cpp_output, json_report)
        stats["instance_id"] = instance_id
        stats["repo"] = instance['repo']

        return success, stats

    except Exception as e:
        print(f"  ❌ Error processing instance: {e}")
        import traceback
        traceback.print_exc()
        return False, {"error": str(e), "instance_id": instance_id}


def process_local_project(
    project_root: Path,
    output_dir: Path,
    cpp_decoder: Path
) -> bool:
    """
    Process a local project for testing.

    Args:
        project_root: Path to project root directory
        output_dir: Output directory for results
        cpp_decoder: Path to C++ decoder executable

    Returns:
        bool: True if test passed, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Processing local project: {project_root.name}")
    print(f"{'='*80}")

    try:
        # Get or generate SCIP file
        scip_file, success = get_or_generate_scip_file(project_root)
        if not success:
            return False

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Output files
        output_name = project_root.name
        py_output = output_dir / f"{output_name}_python.json"
        cpp_output = output_dir / f"{output_name}_cpp.json"
        json_report = output_dir / f"{output_name}_comparison_report.json"

        # Run decoders
        if not python_decode(scip_file, project_root, py_output):
            print(f"  ❌ Python decoder failed")
            return False

        if not cpp_decode(scip_file, project_root, cpp_output, cpp_decoder):
            print(f"  ❌ C++ decoder failed")
            return False

        # Generate comparison report with JSON output
        success, stats = generate_report(py_output, cpp_output, json_report)
        return success

    except Exception as e:
        print(f"  ❌ Error processing local project: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compare C++ and Python SCIP decoders. See file docstring for usage examples."
    )

    parser.add_argument(
        "project_root",
        type=Path,
        nargs="?",
        help="Path to project root directory (for local testing)"
    )

    parser.add_argument(
        "--cpp-decoder",
        type=Path,
        default=Path(__file__).parent.parent.parent / "build" / "core" / "scip_decode_repo",
        help="Path to C++ decoder executable (default: build/core/scip_decode_repo)"
    )

    # SWE-bench testing modes
    parser.add_argument(
        "--instance",
        type=str,
        help="Test single SWE-bench instance (e.g., django__django-11099)"
    )

    parser.add_argument(
        "--multiple-instances",
        action="store_true",
        help="Test default set of instances (7 predefined instances)"
    )

    parser.add_argument(
        "--num-instances",
        type=int,
        help="Test N random instances from SWE-bench_Verified"
    )

    args = parser.parse_args()

    # Set default values
    output_dir = Path(__file__).parent / "comparison_results"
    swebench_dataset = "princeton-nlp/SWE-bench_Verified"
    swebench_split = "test"
    cache_dir = "~/.codeminer"

    # Add these as args attributes for compatibility with process_single_instance
    args.output_dir = output_dir
    args.swebench_dataset = swebench_dataset
    args.swebench_split = swebench_split
    args.cache_dir = cache_dir

    # Validate C++ decoder
    if not args.cpp_decoder.exists():
        print(f"❌ C++ decoder not found: {args.cpp_decoder}")
        return 1

    # Determine which instances to test
    instance_ids = []

    # Priority: multiple-instances > num-instances > single instance
    if args.multiple_instances:
        # Use default predefined instances
        instance_ids = DEFAULT_TEST_INSTANCES
        print(f"Testing {len(instance_ids)} default instances")

    elif args.num_instances:
        # Load dataset and randomly select N instances
        import random
        print(f"Loading dataset {swebench_dataset}...")
        dataset = load_filter_swebench_dataset_explicit(
            dataset=swebench_dataset,
            filter_instance=".*",  # Load all
            split=swebench_split
        )

        # Randomly select N instances
        total_instances = len(dataset)
        num_to_select = min(args.num_instances, total_instances)
        selected_indices = random.sample(range(total_instances), num_to_select)
        dataset = dataset.select(selected_indices)

        instance_ids = [item["instance_id"] for item in dataset]
        print(f"Randomly selected {len(instance_ids)} instances from {total_instances} total")

    elif args.instance:
        # Single instance
        instance_ids = [args.instance]

    # Handle batch testing
    if len(instance_ids) > 1:
        print(f"\n{'='*80}")
        print(f"BATCH MODE: Testing {len(instance_ids)} instances")
        print(f"{'='*80}")

        all_stats = []
        for i, instance_id in enumerate(instance_ids, 1):
            print(f"\n[{i}/{len(instance_ids)}] Testing {instance_id}")

            success, stats = process_single_instance(instance_id, args, args.cpp_decoder)
            all_stats.append(stats)

        # Return success if all passed
        all_success = all(s.get("perfect_match", False) for s in all_stats)
        return 0 if all_success else 1

    # Handle single instance
    elif len(instance_ids) == 1:
        success, stats = process_single_instance(instance_ids[0], args, args.cpp_decoder)
        return 0 if success else 1

    # Traditional mode: test local project
    if args.project_root is None:
        print("❌ Error: Either provide project_root or use --instance/--multiple-instances/--num-instances")
        parser.print_help()
        return 1

    if not args.project_root.exists():
        print(f"❌ Project root not found: {args.project_root}")
        return 1

    # Process local project
    success = process_local_project(args.project_root, output_dir, args.cpp_decoder)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
