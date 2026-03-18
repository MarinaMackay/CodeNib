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
    # Local mode - test with local repository
    python test/graph/test_codegraph_multi.py --lang cpp --repo \
        test/scip/simple_repos/cpp_simple
    python test/graph/test_codegraph_multi.py --lang rust --repo \
        test/scip/simple_repos/rust_simple
    python test/graph/test_codegraph_multi.py --lang ts --repo \
        test/scip/simple_repos/typescript_simple

    # SWE-bench_Multilingual mode - test with dataset
    python test/graph/test_codegraph_multi.py --lang cpp --swebench-multilingual \
        --num-instances 5
    python test/graph/test_codegraph_multi.py --lang rust --swebench-multilingual \
        --num-instances 3
    python test/graph/test_codegraph_multi.py --lang ts --swebench-multilingual \
        --num-instances 2

    # Sample mode - test 5 instances from each language
    python test/graph/test_codegraph_multi.py --sample
    python test/graph/test_codegraph_multi.py --sample --num-instances 3 \
        # Test 3 from each language

    # Pytest mode
    pytest test/graph/test_codegraph_multi.py -k test_cpp_multi -q
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pytest

from codeminer.dataset.swebench_multilingual import SwebenchMultilingualDataset
from codeminer.ls_router import LSIndexer

# ==============================================================================
# Language configuration
# ==============================================================================

LANG_CONFIG = {
    "cpp": {
        "display_name": "C++",
        "indexer_name": "clangd",
        "default_repo": "cpp_simple",
        "repo_keywords": [
            "fmtlib/",
            "nlohmann/",
            "jqlang/",
            "redis/",
            "valkey-io/",
            "micropython/",
        ],
        "pipeline_kwargs": {},
    },
    "rust": {
        "display_name": "Rust",
        "indexer_name": "rust-analyzer",
        "default_repo": "rust_simple",
        "repo_keywords": [
            "tokio-rs/",
            "serde-rs/",
            "clap-rs/",
            "rayon-rs/",
            "sharkdp/",
            "nushell/",
        ],
        "pipeline_kwargs": {"exclude_vendored_libraries": True},
    },
    "ts": {
        "display_name": "TypeScript",
        "indexer_name": "scip-typescript",
        "default_repo": "typescript_simple",
        "repo_keywords": [
            "vuejs/",
            "mui/",
            "darkreader/",
            "sveltejs/",
            "axios/",
            "expressjs/",
            "insomnia/",
            "dayjs/",
        ],
        "pipeline_kwargs": {},
    },
    "go": {
        "display_name": "Go",
        "indexer_name": "scip-go",
        "default_repo": "go_simple",
        "repo_keywords": [
            "gin-gonic/",
            "gohugoio/",
            "hashicorp/",
            "prometheus/",
            "caddyserver/",
        ],
        "pipeline_kwargs": {},
    },
}


def _tools_ready(language: str) -> bool:
    """Check if required external tooling is available."""
    if language == "cpp":
        return bool(shutil.which("clangd")) and bool(shutil.which("cmake"))
    if language == "rust":
        return bool(shutil.which("rust-analyzer"))
    if language == "ts":
        return bool(shutil.which("scip-typescript")) or bool(shutil.which("npx"))
    if language == "go":
        return bool(shutil.which("scip-go"))
    return False


# ==============================================================================
# Shared helpers
# ==============================================================================


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
                print(
                    f"  [{i:2d}] {node_type:15s} | {node_name[:50]:50s} @ {file_path}"
                )
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
# Unified test runners
# ==============================================================================


def run_local(lang, repo_path=None):
    """Test CodeGraph generation with a local project.

    Args:
        lang: Language key ("cpp", "rust", "ts")
        repo_path: Path to repo, or None to use default simple_repos
    """
    cfg = LANG_CONFIG[lang]
    display = cfg["display_name"]

    print("=" * 80)
    print(f"{display} SCIP CodeGraph Test")
    print("=" * 80)

    if repo_path:
        project_path = Path(repo_path).absolute()
    else:
        project_path = (
            Path(__file__).parent.parent
            / "scip"
            / "simple_repos"
            / cfg["default_repo"]
        )

    if not project_path.exists():
        print(f"\n❌ Test project not found at {project_path}")
        return False

    print(f"\nProject path: {project_path}")
    output_path = f"./scip_output/{lang}"
    print(f"Output path: {output_path}")

    indexer = LSIndexer(
        str(project_path),
        output_dir=output_path,
        language=lang,
    )

    print("\n" + "=" * 80)
    print(f"Running {display} Pipeline")
    print("=" * 80)

    print("\nRunning pipeline...")
    graph = indexer.run_pipeline(skip_level="graph", **cfg["pipeline_kwargs"])

    if not graph:
        print("\n❌ Failed to generate CodeGraph")
        return False

    stats = analyze_graph(graph)

    json_file = Path(output_path) / "codegraph.json"
    export_graph_to_json(
        graph, json_file, language=display, indexer=cfg["indexer_name"]
    )

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"\nNodes: {stats['total_nodes']}")
    print(f"Edges: {stats['total_edges']}")
    print(f"References: {stats['reference_edges']}")
    print(f"Containments: {stats['containment_edges']}")
    print(f"\n✅ {display} CodeGraph test completed!")
    print(f"\nGraph saved to: {indexer.graph_file}")
    print("=" * 80 + "\n")

    return True


def run_multi(lang, args):
    """Test CodeGraph with SWE-bench_Multilingual dataset.

    Args:
        lang: Language key ("cpp", "rust", "ts")
        args: Namespace with num_instances

    Returns:
        Tuple of (success_count, total_count, failed_instances)
    """
    cfg = LANG_CONFIG[lang]
    display = cfg["display_name"]

    dataset_obj = SwebenchMultilingualDataset(split="test", filter_instance=".*")
    all_dataset = dataset_obj.load()

    # Filter by repo keywords
    instances = [
        inst
        for inst in all_dataset
        if any(kw in inst["repo"] for kw in cfg["repo_keywords"])
    ]

    if not instances:
        print(f"❌ No {display} instances found in dataset")
        return 0, 0, []

    if args.num_instances is not None:
        dataset = instances[: min(args.num_instances, len(instances))]
    else:
        dataset = instances

    print(
        f"\n✓ Found {len(instances)} total {display} instances, testing {len(dataset)}"
    )

    success_count = 0
    failed_instances = []

    for idx, instance in enumerate(dataset):
        instance_id = instance["instance_id"]

        print(f"\n{'=' * 80}")
        print(
            f"Processing {display} instance [{idx + 1}/{len(dataset)}]: {instance_id}"
        )
        print(f"{'=' * 80}")

        try:
            dataset_obj.process_instance(instance)
            repo_path = dataset_obj.get_repo_path(instance)

            output_path = f"./scip_output/{lang}/{instance_id}"

            indexer = LSIndexer(repo_path, output_dir=output_path, language=lang)

            print("\n[Running CodeGraph pipeline...]")
            graph = indexer.run_pipeline(
                skip_level="graph", **cfg["pipeline_kwargs"]
            )

            if not graph:
                print(f"\n❌ Failed to generate CodeGraph for {instance_id}")
                failed_instances.append((instance_id, "CodeGraph generation failed"))
                continue

            stats = analyze_graph(graph)

            json_file = Path(output_path) / "codegraph.json"
            export_graph_to_json(
                graph, json_file, language=display, indexer=cfg["indexer_name"]
            )

            print(f"\n✅ Successfully processed {instance_id}")
            print(f"   Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}")

            success_count += 1

        except Exception as e:
            print(f"\n❌ Error processing {instance_id}: {str(e)}")
            failed_instances.append((instance_id, str(e)))
            continue

    return success_count, len(dataset), failed_instances


# ==============================================================================
# Sample Mode - Test N instances from each language
# ==============================================================================


def run_sample_mode(num_instances=5):
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

    args = argparse.Namespace(num_instances=num_instances)

    results = {}
    all_failed = []

    for lang, cfg in LANG_CONFIG.items():
        display = cfg["display_name"]
        print("\n" + "=" * 80)
        print(f"Testing {display} ({num_instances} instances)")
        print("=" * 80)

        ok, total, failed = run_multi(lang, args)
        results[display] = {
            "success": total > 0 and ok == total,
            "failed_count": len(failed),
        }
        all_failed.extend([(display, f) for f in failed])

    # Print overall summary
    print("\n" + "=" * 80)
    print("Sample Mode Summary")
    print("=" * 80)

    for lang_display, result in results.items():
        status = (
            "✅ PASSED"
            if result["success"]
            else f"❌ FAILED ({result['failed_count']} failures)"
        )
        print(f"\n{lang_display}: {status}")

    if all_failed:
        print(f"\n\nFailed instances across all languages:")
        for lang_display, (instance_id, error) in all_failed:
            error_msg = error[:80] + "..." if len(error) > 80 else error
            print(f"  [{lang_display}] {instance_id}: {error_msg}")

    print("\n" + "=" * 80)

    return all(r["success"] for r in results.values())


# ==============================================================================
# Pytest
# ==============================================================================


@pytest.fixture(scope="session")
def swebench_args() -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(
        ["--lang", "cpp", "--swebench-multilingual", "--num-instances", "1"]
    )


def test_cpp_multi(swebench_args: argparse.Namespace) -> None:
    if not _tools_ready("cpp"):
        pytest.skip("clangd/cmake not available")
    ok, total, failed = run_multi("cpp", swebench_args)
    assert ok > 0, f"All {total} C++ instances failed: {failed}"


def test_rust_multi(swebench_args: argparse.Namespace) -> None:
    if not _tools_ready("rust"):
        pytest.skip("rust-analyzer not available")
    ok, total, failed = run_multi("rust", swebench_args)
    assert ok > 0, f"All {total} Rust instances failed: {failed}"


def test_ts_multi(swebench_args: argparse.Namespace) -> None:
    if not _tools_ready("ts"):
        pytest.skip("scip-typescript/npx not available")
    ok, total, failed = run_multi("ts", swebench_args)
    assert ok > 0, f"All {total} TypeScript instances failed: {failed}"


def test_go_multi(swebench_args: argparse.Namespace) -> None:
    if not _tools_ready("go"):
        pytest.skip("scip-go not available")
    ok, total, failed = run_multi("go", swebench_args)
    assert ok > 0, f"All {total} Go instances failed: {failed}"


def build_parser() -> argparse.ArgumentParser:
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
        "--swebench-multilingual",
        dest="swebench_multilingual",
        action="store_true",
        help="Test with SWE-bench_Multilingual dataset for selected language",
    )
    mode_group.add_argument(
        "--multisweb",
        dest="swebench_multilingual",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    mode_group.add_argument(
        "--repo",
        type=str,
        help="Path to local repository (local mode)",
    )

    # Language selection (not used in sample mode)
    parser.add_argument(
        "--lang",
        choices=["cpp", "rust", "ts", "go"],
        help="Language to test (required for --repo and --swebench-multilingual modes)",
    )

    # Number of instances
    parser.add_argument(
        "--num-instances",
        type=int,
        default=5,
        help="Number of instances to test (default: 5)",
    )

    return parser


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Sample mode - test all languages
    if args.sample:
        print("\n=== Running in Sample mode ===")
        print(f"Testing {args.num_instances} instances from EACH language\n")
        success = run_sample_mode(args.num_instances)
        sys.exit(0 if success else 1)

    # For other modes, language is required
    if not args.lang:
        parser.error("--lang is required when not using --sample mode")

    # SWE-bench_Multilingual mode
    if args.swebench_multilingual:
        display = LANG_CONFIG[args.lang]["display_name"]
        print(f"\n=== Running in SWE-bench_Multilingual mode ({display}) ===\n")
        ok, total, _ = run_multi(args.lang, args)
        sys.exit(0 if ok == total else 1)

    # Local mode
    if args.repo:
        display = LANG_CONFIG[args.lang]["display_name"]
        print(f"\n=== Running in Local mode ({display}) ===\n")
    else:
        display = LANG_CONFIG[args.lang]["display_name"]
        print(f"\n=== Running in Local mode (default, {display}) ===\n")

    success = run_local(args.lang, args.repo)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
