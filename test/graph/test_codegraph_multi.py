#!/usr/bin/env python3
"""
Unified SCIP CodeGraph test for C++, Rust, and TypeScript.

This script tests the complete pipeline for all three languages:
1. Generate SCIP index from project
2. Decode the binary index
3. Parse into CodeGraph
4. Analyze the graph structure
5. Export to JSON for manual verification

Usage:
    # Simple mode - test with local repository
    python test/graph/test_codegraph_multi.py --lang cpp --repo test/scip/simple_repos/cpp_simple
    python test/graph/test_codegraph_multi.py --lang rust --repo test/scip/simple_repos/rust_simple
    python test/graph/test_codegraph_multi.py --lang ts --repo test/scip/simple_repos/typescript_simple

    # Multi-SWE-bench mode - test with dataset
    python test/graph/test_codegraph_multi.py --lang cpp --multisweb --num-instances 5
    python test/graph/test_codegraph_multi.py --lang rust --multisweb --num-instances 3
    python test/graph/test_codegraph_multi.py --lang ts --multisweb --num-instances 2

    # Sample mode - test 5 instances from each language
    python test/graph/test_codegraph_multi.py --sample
    python test/graph/test_codegraph_multi.py --sample --num-instances 3  # Test 3 from each language
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from codeminer.env import load_filter_swebench_dataset, process_swebench_instance
from codeminer.scip_interface import SCIPClangIndexer, SCIPRustIndexer, SCIPTypeScriptIndexer


def analyze_graph(graph):
    """Analyze and print statistics about the generated graph."""
    print("\n" + "=" * 80)
    print("CodeGraph Analysis")
    print("=" * 80)

    # Get graph object
    ig_graph = graph["graph"] if isinstance(graph, dict) else graph.graph

    # Count nodes by type
    node_types = {}
    for node in ig_graph.vs:
        node_type = node["type"]
        node_types[node_type] = node_types.get(node_type, 0) + 1

    print(f"\nTotal nodes: {len(ig_graph.vs)}")
    print("\nNode types:")
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type:20s}: {count:5d}")

    # Count edges by type
    edge_types = {}
    for edge in ig_graph.es:
        edge_type = edge["type"]
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    print(f"\nTotal edges: {len(ig_graph.es)}")
    print("\nEdge types:")
    for edge_type, count in sorted(edge_types.items()):
        print(f"  {edge_type:20s}: {count:5d}")

    # Sample some nodes
    print("\n" + "=" * 80)
    print("Sample Nodes (first 30)")
    print("=" * 80)

    for i, node in enumerate(ig_graph.vs[:30]):
        node_name = node["name"]
        node_type = node["type"]
        # Show file path for symbols
        if "file" in node.attributes():
            file_path = node["file"]
            if file_path and file_path != node_name:
                print(f"  [{i:2d}] {node_type:15s} | {node_name[:50]:50s} @ {file_path}")
            else:
                print(f"  [{i:2d}] {node_type:15s} | {node_name[:50]}")
        else:
            print(f"  [{i:2d}] {node_type:15s} | {node_name[:50]}")

    # Find and display reference edges
    print("\n" + "=" * 80)
    print("Sample Reference Edges (first 15)")
    print("=" * 80)

    ref_edges = [e for e in ig_graph.es if e["type"] == "reference"]
    print(f"\nTotal reference edges: {len(ref_edges)}")

    for i, edge in enumerate(ref_edges[:15]):
        source = ig_graph.vs[edge.source]["name"]
        target = ig_graph.vs[edge.target]["name"]
        print(f"  [{i:2d}] {source[:40]:40s} -> {target[:40]:40s}")

    # Find and display containment edges
    print("\n" + "=" * 80)
    print("Sample Containment Edges (first 15)")
    print("=" * 80)

    contain_edges = [e for e in ig_graph.es if e["type"] == "contain"]
    print(f"\nTotal containment edges: {len(contain_edges)}")

    for i, edge in enumerate(contain_edges[:15]):
        source = ig_graph.vs[edge.source]["name"]
        target = ig_graph.vs[edge.target]["name"]
        print(f"  [{i:2d}] {source[:40]:40s} -> {target[:40]:40s}")

    return {
        "total_nodes": len(ig_graph.vs),
        "total_edges": len(ig_graph.es),
        "node_types": node_types,
        "edge_types": edge_types,
        "reference_edges": len(ref_edges),
        "containment_edges": len(contain_edges),
    }


def export_graph_to_json(graph, output_file, language, indexer):
    """
    Export CodeGraph to JSON file for manual verification.

    Args:
        graph: CodeGraph object or dict containing graph
        output_file: Output JSON file path
        language: Language name (C++, Rust, TypeScript)
        indexer: Indexer name (scip-clang, rust-analyzer, scip-typescript)
    """
    ig_graph = graph["graph"] if isinstance(graph, dict) else graph.graph

    # Extract nodes
    nodes = []
    for v in ig_graph.vs:
        node = {"id": v.index}

        # Safe attribute access
        try:
            node["name"] = v["name"]
        except KeyError:
            node["name"] = f"node_{v.index}"

        try:
            node["type"] = v["type"]
        except KeyError:
            node["type"] = "unknown"

        # Add optional attributes
        for attr in ["scip_symbol", "display_name", "file", "start_line", "end_line"]:
            try:
                node[attr] = v[attr]
            except KeyError:
                pass

        nodes.append(node)

    # Extract edges
    edges = []
    for e in ig_graph.es:
        try:
            edge = {
                "source": ig_graph.vs[e.source]["name"],
                "target": ig_graph.vs[e.target]["name"],
                "type": e["type"],
            }
            edges.append(edge)
        except KeyError:
            pass

    # Compute statistics
    node_type_counts = {}
    for node in nodes:
        node_type = node["type"]
        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

    edge_type_counts = {}
    for edge in edges:
        edge_type = edge["type"]
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1

    # Create node samples (first 5 of each type)
    node_samples = {}
    for node_type in node_type_counts:
        samples = [n for n in nodes if n["type"] == node_type][:5]
        node_samples[node_type] = samples

    # Create edge samples (first 10 of each type)
    edge_samples = {}
    for edge_type in edge_type_counts:
        samples = [e for e in edges if e["type"] == edge_type][:10]
        edge_samples[edge_type] = samples

    # Build complete data structure
    graph_dict = {
        "metadata": {
            "vertex_count": ig_graph.vcount(),
            "edge_count": ig_graph.ecount(),
            "node_type_counts": node_type_counts,
            "edge_type_counts": edge_type_counts,
        },
        "samples": {
            "nodes": node_samples,
            "edges": edge_samples,
        },
        "full_data": {
            "nodes": nodes,
            "edges": edges,
        },
        "source": {
            "language": language,
            "indexer": indexer,
        },
    }

    # Save as JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_dict, f, indent=2, ensure_ascii=False)

    print(f"\n✅ CodeGraph exported to JSON: {output_path}")
    print(f"   Nodes: {graph_dict['metadata']['vertex_count']}")
    print(f"   Edges: {graph_dict['metadata']['edge_count']}")
    print(f"   Node types: {node_type_counts}")
    print(f"   Edge types: {edge_type_counts}")

    return output_path


# ==============================================================================
# C++ Tests
# ==============================================================================

def test_cpp_simple(repo_path=None):
    """Test C++ CodeGraph generation with simple project."""
    print("=" * 80)
    print("C++ SCIP CodeGraph Test")
    print("=" * 80)

    # Use provided path or default to cpp_simple
    if repo_path:
        project_path = Path(repo_path).absolute()
    else:
        project_path = (
            Path(__file__).parent.parent / "scip" / "simple_repos" / "cpp_simple"
        )

    if not project_path.exists():
        print(f"\n❌ Test project not found at {project_path}")
        return False

    # Check for CMakeLists.txt
    if not (project_path / "CMakeLists.txt").exists():
        print(f"\n❌ No CMakeLists.txt found at {project_path}")
        return False

    print(f"\nProject path: {project_path}")
    output_path = "./scip_output/cpp"
    print(f"Output path: {output_path}")

    # Create C++ indexer
    indexer = SCIPClangIndexer(
        str(project_path),
        output_dir=output_path,
    )

    print("\n" + "=" * 80)
    print("Running SCIP C++ Pipeline")
    print("=" * 80)

    # Run the complete pipeline
    print("\nRunning pipeline...")
    graph = indexer.run_pipeline(skip_level="graph")

    if not graph:
        print("\n❌ Failed to generate CodeGraph")
        return False

    # Analyze the graph
    stats = analyze_graph(graph)

    # Export to JSON
    json_file = Path(output_path) / "codegraph.json"
    export_graph_to_json(graph, json_file, language="C++", indexer="scip-clang")

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"\nNodes: {stats['total_nodes']}")
    print(f"Edges: {stats['total_edges']}")
    print(f"References: {stats['reference_edges']}")
    print(f"Containments: {stats['containment_edges']}")
    print(f"\n✅ C++ CodeGraph test completed!")
    print(f"\nGraph saved to: {indexer.graph_file}")
    print("=" * 80 + "\n")

    return True


def test_cpp_multisweb(args):
    """Test C++ SCIP CodeGraph with Multi-SWE-bench dataset."""
    # SWE-bench Multilingual dataset configuration
    args_dict = {
        "dataset": "SWE-bench/SWE-bench_Multilingual",
        "split": "test",
        "filter_instance": ".*",
    }

    dataset_args = argparse.Namespace(**args_dict)
    all_dataset = load_filter_swebench_dataset(args=dataset_args)

    # Filter for C/C++ projects
    cpp_instances = []
    cpp_repo_keywords = ['fmtlib/', 'nlohmann/', 'jqlang/', 'redis/', 'valkey-io/', 'micropython/']

    for inst in all_dataset:
        repo = inst['repo']
        if any(keyword in repo for keyword in cpp_repo_keywords):
            cpp_instances.append(inst)

    if len(cpp_instances) == 0:
        print(f"❌ No C/C++ instances found in dataset")
        return False, []

    # Limit dataset to specified number of instances
    if args.num_instances is not None:
        dataset = cpp_instances[: min(args.num_instances, len(cpp_instances))]
    else:
        dataset = cpp_instances

    print(f"\n✓ Found {len(cpp_instances)} total C/C++ instances, testing {len(dataset)}")

    # Statistics tracking
    success_count = 0
    failed_instances = []

    # Process each instance
    for idx, instance in enumerate(dataset):
        instance_id = instance["instance_id"]

        print(f"\n{'=' * 80}")
        print(f"Processing C++ instance [{idx + 1}/{len(dataset)}]: {instance_id}")
        print(f"{'=' * 80}")

        try:
            # Download and checkout repository
            repo_path = process_swebench_instance(instance)

            # Set output path
            output_path = f"./scip_output/cpp/{instance_id}"

            # Check for CMakeLists.txt or Makefile
            cmake_file = Path(repo_path) / "CMakeLists.txt"
            makefile = Path(repo_path) / "Makefile"

            if cmake_file.exists():
                # Try to generate compilation database
                build_dir = Path(repo_path) / "build"
                build_dir.mkdir(exist_ok=True)

                result = subprocess.run(
                    ["cmake", "-B", "build", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    print(f"\n❌ CMake configuration failed")
                    failed_instances.append((instance_id, "CMake configuration failed"))
                    continue

            # Create C++ indexer
            indexer = SCIPClangIndexer(repo_path, output_dir=output_path)

            # Run complete pipeline
            print("\n[Running CodeGraph pipeline...]")
            graph = indexer.run_pipeline(skip_level="graph")

            if not graph:
                print(f"\n❌ Failed to generate CodeGraph for {instance_id}")
                failed_instances.append((instance_id, "CodeGraph generation failed"))
                continue

            # Analyze the graph
            stats = analyze_graph(graph)

            # Export to JSON
            json_file = Path(output_path) / "codegraph.json"
            export_graph_to_json(graph, json_file, language="C++", indexer="scip-clang")

            print(f"\n✅ Successfully processed {instance_id}")
            print(f"   Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}")

            success_count += 1

        except Exception as e:
            print(f"\n❌ Error processing {instance_id}: {str(e)}")
            failed_instances.append((instance_id, str(e)))
            continue

    return success_count == len(dataset), failed_instances


# ==============================================================================
# Rust Tests
# ==============================================================================

def test_rust_simple(repo_path=None):
    """Test Rust CodeGraph generation with simple project."""
    print("=" * 80)
    print("Rust SCIP CodeGraph Test")
    print("=" * 80)

    # Use provided path or default to rust_simple
    if repo_path:
        project_path = Path(repo_path).absolute()
    else:
        project_path = (
            Path(__file__).parent.parent / "scip" / "simple_repos" / "rust_simple"
        )

    if not project_path.exists():
        print(f"\n❌ Test project not found at {project_path}")
        return False

    # Check for Cargo.toml
    if not (project_path / "Cargo.toml").exists():
        print(f"\n❌ No Cargo.toml found at {project_path}")
        return False

    print(f"\nProject path: {project_path}")
    output_path = "./scip_output/rust"
    print(f"Output path: {output_path}")

    # Create Rust indexer
    indexer = SCIPRustIndexer(
        str(project_path),
        output_dir=output_path,
    )

    print("\n" + "=" * 80)
    print("Running SCIP Rust Pipeline")
    print("=" * 80)

    # Run the complete pipeline
    print("\nRunning pipeline...")
    graph = indexer.run_pipeline(
        exclude_vendored_libraries=True,
        skip_level="graph",
    )

    if not graph:
        print("\n❌ Failed to generate CodeGraph")
        return False

    # Analyze the graph
    stats = analyze_graph(graph)

    # Export to JSON
    json_file = Path(output_path) / "codegraph.json"
    export_graph_to_json(graph, json_file, language="Rust", indexer="rust-analyzer")

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"\nNodes: {stats['total_nodes']}")
    print(f"Edges: {stats['total_edges']}")
    print(f"References: {stats['reference_edges']}")
    print(f"Containments: {stats['containment_edges']}")
    print(f"\n✅ Rust CodeGraph test completed!")
    print(f"\nGraph saved to: {indexer.graph_file}")
    print("=" * 80 + "\n")

    return True


def test_rust_multisweb(args):
    """Test Rust SCIP CodeGraph with Multi-SWE-bench dataset."""
    # SWE-bench Multilingual dataset configuration
    args_dict = {
        "dataset": "SWE-bench/SWE-bench_Multilingual",
        "split": "test",
        "filter_instance": ".*",
    }

    dataset_args = argparse.Namespace(**args_dict)
    all_dataset = load_filter_swebench_dataset(args=dataset_args)

    # Filter for Rust projects only
    rust_instances = []
    rust_repo_keywords = ['tokio-rs/', 'serde-rs/', 'clap-rs/', 'rayon-rs/',
                          'sharkdp/', 'nushell/']

    for inst in all_dataset:
        repo = inst['repo']
        if any(keyword in repo for keyword in rust_repo_keywords):
            rust_instances.append(inst)

    if len(rust_instances) == 0:
        print(f"❌ No Rust instances found in dataset")
        return False, []

    # Limit dataset to specified number of instances
    if args.num_instances is not None:
        dataset = rust_instances[: min(args.num_instances, len(rust_instances))]
    else:
        dataset = rust_instances

    print(f"\n✓ Found {len(rust_instances)} total Rust instances, testing {len(dataset)}")

    # Statistics tracking
    success_count = 0
    failed_instances = []

    # Process each instance
    for idx, instance in enumerate(dataset):
        instance_id = instance["instance_id"]

        print(f"\n{'=' * 80}")
        print(f"Processing Rust instance [{idx + 1}/{len(dataset)}]: {instance_id}")
        print(f"{'=' * 80}")

        try:
            # Download and checkout repository
            repo_path = process_swebench_instance(instance)

            # Set output path
            output_path = f"./scip_output/rust/{instance_id}"

            # Check if Cargo.toml exists
            cargo_toml = Path(repo_path) / "Cargo.toml"
            if not cargo_toml.exists():
                print(f"\n⚠️  No Cargo.toml found, skipping...")
                failed_instances.append((instance_id, "No Cargo.toml found"))
                continue

            # Create Rust indexer
            indexer = SCIPRustIndexer(repo_path, output_dir=output_path)

            # Run complete pipeline
            print("\n[Running CodeGraph pipeline...]")
            graph = indexer.run_pipeline(
                exclude_vendored_libraries=True,
                skip_level="graph",
            )

            if not graph:
                print(f"\n❌ Failed to generate CodeGraph for {instance_id}")
                failed_instances.append((instance_id, "CodeGraph generation failed"))
                continue

            # Analyze the graph
            stats = analyze_graph(graph)

            # Export to JSON
            json_file = Path(output_path) / "codegraph.json"
            export_graph_to_json(graph, json_file, language="Rust", indexer="rust-analyzer")

            print(f"\n✅ Successfully processed {instance_id}")
            print(f"   Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}")

            success_count += 1

        except Exception as e:
            print(f"\n❌ Error processing {instance_id}: {str(e)}")
            failed_instances.append((instance_id, str(e)))
            continue

    return success_count == len(dataset), failed_instances


# ==============================================================================
# TypeScript Tests
# ==============================================================================

def test_ts_simple(repo_path=None):
    """Test TypeScript CodeGraph generation with simple project."""
    print("=" * 80)
    print("TypeScript SCIP CodeGraph Test")
    print("=" * 80)

    # Use provided path or default to typescript_simple
    if repo_path:
        project_path = Path(repo_path).absolute()
    else:
        project_path = Path(__file__).parent.parent / "scip" / "simple_repos" / "typescript_simple"

    if not project_path.exists():
        print(f"\n❌ Test project not found at {project_path}")
        return False

    print(f"\nProject path: {project_path}")
    output_path = "./scip_output/ts"
    print(f"Output path: {output_path}")

    # Create TypeScript indexer
    indexer = SCIPTypeScriptIndexer(
        str(project_path),
        output_dir=output_path,
    )

    print("\n" + "=" * 80)
    print("Running SCIP TypeScript Pipeline")
    print("=" * 80)

    # Run the complete pipeline
    print("\nRunning pipeline...")
    graph = indexer.run_pipeline(
        infer_tsconfig=False,
        skip_level="graph",
    )

    if not graph:
        print("\n❌ Failed to generate CodeGraph")
        return False

    # Analyze the graph
    stats = analyze_graph(graph)

    # Export to JSON
    json_file = Path(output_path) / "codegraph.json"
    export_graph_to_json(graph, json_file, language="TypeScript", indexer="scip-typescript")

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"\nNodes: {stats['total_nodes']}")
    print(f"Edges: {stats['total_edges']}")
    print(f"References: {stats['reference_edges']}")
    print(f"Containments: {stats['containment_edges']}")
    print(f"\n✅ TypeScript CodeGraph test completed!")
    print(f"\nGraph saved to: {indexer.graph_file}")
    print("=" * 80 + "\n")

    return True


def test_ts_multisweb(args):
    """Test TypeScript SCIP CodeGraph with Multi-SWE-bench dataset."""
    # Load dataset
    dataset_args_dict = {
        "dataset": "SWE-bench/SWE-bench_Multilingual",
        "split": "test",
        "filter_instance": ".*",
    }
    dataset_args = argparse.Namespace(**dataset_args_dict)
    all_dataset = load_filter_swebench_dataset(args=dataset_args)

    # Filter for TypeScript/JavaScript projects
    ts_instances = []
    ts_repo_keywords = [
        "vuejs/",
        "mui/",
        "darkreader/",
        "sveltejs/",
        "axios/",
        "expressjs/",
        "insomnia/",
        "dayjs/",
    ]

    for inst in all_dataset:
        repo = inst["repo"]
        if any(keyword in repo for keyword in ts_repo_keywords):
            ts_instances.append(inst)

    if len(ts_instances) == 0:
        print(f"❌ No TypeScript instances found in dataset")
        return False, []

    # Limit to requested number
    if args.num_instances is not None:
        dataset = ts_instances[: min(args.num_instances, len(ts_instances))]
    else:
        dataset = ts_instances

    print(f"\n✓ Found {len(ts_instances)} total TypeScript instances, testing {len(dataset)}")

    # Statistics
    success_count = 0
    failed_instances = []

    # Process each instance
    for idx, instance in enumerate(dataset):
        instance_id = instance["instance_id"]
        print(f"\n{'=' * 80}")
        print(f"Processing TypeScript instance [{idx + 1}/{len(dataset)}]: {instance_id}")
        print(f"{'=' * 80}")

        try:
            # Download repository
            repo_path = process_swebench_instance(instance)

            # Set output path
            output_path = f"./scip_output/ts/{instance_id}"

            # Create indexer
            indexer = SCIPTypeScriptIndexer(repo_path, output_dir=output_path)

            # Check if tsconfig.json exists
            tsconfig = Path(repo_path) / "tsconfig.json"
            infer_tsconfig = not tsconfig.exists()

            if infer_tsconfig:
                print("  ⚠️  No tsconfig.json found, will infer TypeScript configuration")

            # Install dependencies if package.json exists
            package_json = Path(repo_path) / "package.json"
            if package_json.exists():
                print("  Installing dependencies...")
                try:
                    result = subprocess.run(
                        ["npm", "install"],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if result.returncode != 0:
                        print(f"  ⚠️  npm install failed: {result.stderr[:200]}")
                except subprocess.TimeoutExpired:
                    print("  ⚠️  npm install timed out (5 min)")
                except Exception as e:
                    print(f"  ⚠️  npm install error: {e}")

            # Run pipeline
            print("\n[Running CodeGraph pipeline...]")
            graph = indexer.run_pipeline(
                skip_level="graph",
                infer_tsconfig=infer_tsconfig,
            )

            if not graph:
                print(f"❌ Failed to generate CodeGraph for {instance_id}")
                failed_instances.append((instance_id, "CodeGraph generation failed"))
                continue

            # Analyze graph
            stats = analyze_graph(graph)

            # Export to JSON
            json_file = Path(output_path) / "codegraph.json"
            export_graph_to_json(graph, json_file, language="TypeScript", indexer="scip-typescript")

            print(f"\n✅ Successfully processed {instance_id}")
            print(f"   Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}")

            success_count += 1

        except Exception as e:
            print(f"\n❌ Error processing {instance_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_instances.append((instance_id, str(e)))
            continue

    return success_count == len(dataset), failed_instances


# ==============================================================================
# Sample Mode - Test N instances from each language
# ==============================================================================

def test_sample_mode(num_instances=5):
    """
    Test N instances from EACH language (C++, Rust, TypeScript).

    Args:
        num_instances: Number of instances to test from each language

    Returns:
        bool: True if all tests passed
    """
    print("\n" + "=" * 80)
    print(f"Sample Mode: Testing {num_instances} instances from each language")
    print("=" * 80)

    # Create args object for each language test
    args = argparse.Namespace(num_instances=num_instances)

    results = {}
    all_failed = []

    # Test C++
    print("\n" + "=" * 80)
    print(f"Testing C++ ({num_instances} instances)")
    print("=" * 80)
    cpp_success, cpp_failed = test_cpp_multisweb(args)
    results["C++"] = {"success": cpp_success, "failed_count": len(cpp_failed)}
    all_failed.extend([("C++", f) for f in cpp_failed])

    # Test Rust
    print("\n" + "=" * 80)
    print(f"Testing Rust ({num_instances} instances)")
    print("=" * 80)
    rust_success, rust_failed = test_rust_multisweb(args)
    results["Rust"] = {"success": rust_success, "failed_count": len(rust_failed)}
    all_failed.extend([("Rust", f) for f in rust_failed])

    # Test TypeScript
    print("\n" + "=" * 80)
    print(f"Testing TypeScript ({num_instances} instances)")
    print("=" * 80)
    ts_success, ts_failed = test_ts_multisweb(args)
    results["TypeScript"] = {"success": ts_success, "failed_count": len(ts_failed)}
    all_failed.extend([("TypeScript", f) for f in ts_failed])

    # Print overall summary
    print("\n" + "=" * 80)
    print("Sample Mode Summary")
    print("=" * 80)

    for lang, result in results.items():
        status = "✅ PASSED" if result["success"] else f"❌ FAILED ({result['failed_count']} failures)"
        print(f"\n{lang}: {status}")

    if all_failed:
        print(f"\n\nFailed instances across all languages:")
        for lang, (instance_id, error) in all_failed:
            error_msg = error[:80] + "..." if len(error) > 80 else error
            print(f"  [{lang}] {instance_id}: {error_msg}")

    print("\n" + "=" * 80)

    # Return true only if all languages passed
    return all(r["success"] for r in results.values())


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified SCIP CodeGraph test for C++, Rust, and TypeScript"
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--sample",
        action="store_true",
        help="Test N instances from EACH language (C++, Rust, TypeScript)",
    )
    mode_group.add_argument(
        "--multisweb",
        action="store_true",
        help="Test with Multi-SWE-bench dataset for selected language",
    )
    mode_group.add_argument(
        "--repo",
        type=str,
        help="Path to local repository (simple mode)",
    )

    # Language selection (not used in sample mode)
    parser.add_argument(
        "--lang",
        choices=["cpp", "rust", "ts"],
        help="Language to test (required for --repo and --multisweb modes)",
    )

    # Number of instances
    parser.add_argument(
        "--num-instances",
        type=int,
        default=5,
        help="Number of instances to test (default: 5)",
    )

    args = parser.parse_args()

    # Sample mode - test all languages
    if args.sample:
        print("\n=== Running in Sample mode ===")
        print(f"Testing {args.num_instances} instances from EACH language\n")
        success = test_sample_mode(args.num_instances)
        sys.exit(0 if success else 1)

    # For other modes, language is required
    if not args.lang:
        parser.error("--lang is required when not using --sample mode")

    # Multi-SWE-bench mode
    if args.multisweb:
        print(f"\n=== Running in Multi-SWE-bench mode ({args.lang.upper()}) ===\n")
        if args.lang == "cpp":
            success, _ = test_cpp_multisweb(args)
        elif args.lang == "rust":
            success, _ = test_rust_multisweb(args)
        elif args.lang == "ts":
            success, _ = test_ts_multisweb(args)
        sys.exit(0 if success else 1)

    # Simple mode with custom repo
    if args.repo:
        print(f"\n=== Running in Simple mode ({args.lang.upper()}) ===\n")
        if args.lang == "cpp":
            success = test_cpp_simple(args.repo)
        elif args.lang == "rust":
            success = test_rust_simple(args.repo)
        elif args.lang == "ts":
            success = test_ts_simple(args.repo)
        sys.exit(0 if success else 1)

    # Default: simple mode with default repo
    print(f"\n=== Running in Simple mode (default, {args.lang.upper()}) ===\n")
    if args.lang == "cpp":
        success = test_cpp_simple(None)
    elif args.lang == "rust":
        success = test_rust_simple(None)
    elif args.lang == "ts":
        success = test_ts_simple(None)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
