#!/usr/bin/env python3
"""
Unified SCIP indexer test for C++, Rust, and TypeScript.

This test supports indexing and decoding SCIP indexes for all three languages.
It supports two modes:
1. Simple mode: Test with a local repository
2. Multi-SWE-bench mode: Test with SWE-bench Multilingual dataset

Usage:
    # Simple mode - C++
    python test/scip/test_scip_multilingual.py --lang cpp --repo test/scip/simple_repos/cpp_simple
    python test/scip/test_scip_multilingual.py --lang cpp --repo test/scip/simple_repos/cpp_simple --clean

    # Simple mode - Rust
    python test/scip/test_scip_multilingual.py --lang rust --repo test/scip/simple_repos/rust_simple
    python test/scip/test_scip_multilingual.py --lang rust --repo test/scip/simple_repos/rust_simple --clean

    # Simple mode - TypeScript
    python test/scip/test_scip_multilingual.py --lang ts --repo test/scip/simple_repos/typescript_simple

    # Multi-SWE-bench mode
    python test/scip/test_scip_multilingual.py --lang cpp --multisweb --num-instances 10
    python test/scip/test_scip_multilingual.py --lang rust --multisweb --num-instances 5
    python test/scip/test_scip_multilingual.py --lang ts --multisweb --num-instances 3
"""

import argparse
import json
import subprocess
from pathlib import Path

from codeminer.scip_interface import SCIPIndexer


# ==============================================================================
# Common Utilities
# ==============================================================================


def verify_scip_output(decoded_file: Path, language: str, expected_symbols=None) -> bool:
    """
    Verify the SCIP output contains expected symbols.

    Args:
        decoded_file: Path to the decoded SCIP file
        language: Language (C++, Rust, TypeScript)
        expected_symbols: List of symbols to check for (optional)

    Returns:
        bool: True if verification passed
    """
    print("\n" + "=" * 80)
    print("Verifying SCIP output...")
    print("=" * 80)

    if not decoded_file.exists():
        print(f"❌ Decoded file not found: {decoded_file}")
        return False

    content = decoded_file.read_text()

    # Default expected symbols per language
    if expected_symbols is None:
        if language == "C++":
            expected_symbols = [
                "MathUtils",  # Namespace
                "add",  # Function
                "Shape",  # Base class
                "Rectangle",  # Derived class
                "Circle",  # Derived class
            ]
        elif language == "Rust":
            expected_symbols = [
                "math_utils",  # Module
                "add",  # Function
                "multiply",  # Function
                "Shape",  # Trait
                "Rectangle",  # Struct
                "Circle",  # Struct
            ]
        elif language == "TypeScript":
            expected_symbols = [
                "add",  # Function
                "multiply",  # Function
                "Calculator",  # Class
            ]

    if not expected_symbols:
        print("No symbols specified for verification, skipping...")
        return True

    print("\nChecking for expected symbols:")
    found_symbols = []
    missing_symbols = []

    for symbol in expected_symbols:
        if symbol in content:
            found_symbols.append(symbol)
            print(f"  ✅ Found: {symbol}")
        else:
            missing_symbols.append(symbol)
            print(f"  ⚠️  Missing: {symbol}")

    print(f"\nSummary: {len(found_symbols)}/{len(expected_symbols)} symbols found")

    return len(found_symbols) > 0


# ==============================================================================
# C++ Build and Indexing
# ==============================================================================


def build_cpp_project(repo_path: Path, clean: bool = False) -> bool:
    """
    Build the C++ project and generate compile_commands.json using CMake.

    Args:
        repo_path: Path to the repository
        clean: Whether to clean before building

    Returns:
        bool: True if build was successful
    """
    build_dir = repo_path / "build"

    if clean and build_dir.exists():
        print(f"Cleaning build directory: {build_dir}")
        import shutil

        shutil.rmtree(build_dir)

    # Create build directory
    build_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("Building C++ project with CMake...")
    print("=" * 80)

    # Run cmake to generate build files
    print("\nRunning CMake...")
    result = subprocess.run(
        [
            "cmake",
            "-B",
            str(build_dir),
            "-S",
            str(repo_path),
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ CMake failed:")
        print(result.stderr)
        return False

    print("✅ CMake configuration successful")

    # Build the project
    print("\nBuilding project...")
    result = subprocess.run(
        ["cmake", "--build", str(build_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ Build failed:")
        print(result.stderr)
        return False

    print("✅ Build successful")

    # Check if compile_commands.json was generated
    compdb = build_dir / "compile_commands.json"
    if not compdb.exists():
        print(f"❌ compile_commands.json not found at {compdb}")
        return False

    print(f"✅ Generated compilation database at: {compdb}")
    return True


def generate_cpp_compilation_database(repo_path: Path) -> bool:
    """
    Try to generate a compilation database for C++ repository.

    Args:
        repo_path: Path to the repository root

    Returns:
        bool: True if compilation database was generated or already exists
    """
    # Check if compilation database already exists
    compdb_locations = [
        repo_path / "compile_commands.json",
        repo_path / "build" / "compile_commands.json",
    ]

    for compdb in compdb_locations:
        if compdb.exists():
            print(f"✅ Found existing compilation database: {compdb}")
            return True

    print("\n🔧 Attempting to generate compilation database...")

    # Try CMake if CMakeLists.txt exists
    if (repo_path / "CMakeLists.txt").exists():
        print("  Found CMakeLists.txt, trying CMake...")
        try:
            build_dir = repo_path / "build"
            build_dir.mkdir(exist_ok=True)

            result = subprocess.run(
                [
                    "cmake",
                    "-B",
                    str(build_dir),
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                ],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                compdb = build_dir / "compile_commands.json"
                if compdb.exists():
                    print(f"  ✅ Generated compilation database with CMake: {compdb}")
                    return True

        except Exception as e:
            print(f"  ⚠️  CMake error: {str(e)[:200]}")

    print("  ❌ Could not generate compilation database")
    return False


def test_cpp_simple(args):
    """Test C++ SCIP indexer with a local repository."""
    repo_path = Path(args.repo).absolute()

    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        return False

    print("=" * 80)
    print("SCIP C++ Indexer Test - Simple Mode")
    print("=" * 80)
    print(f"Repository path: {repo_path}")
    print("=" * 80)

    # Build the project if CMakeLists.txt exists
    if (repo_path / "CMakeLists.txt").exists():
        print("\n📦 Found CMakeLists.txt, building project...")
        if not build_cpp_project(repo_path, clean=args.clean):
            print("\n❌ Build failed")
            return False
    else:
        print("\n⚠️  No CMakeLists.txt found, skipping build")

    # Check for compilation database
    compdb = repo_path / "compile_commands.json"
    if not compdb.exists():
        compdb = repo_path / "build" / "compile_commands.json"

    if not compdb.exists():
        print(f"\n❌ Error: Compilation database not found!")
        return False

    print(f"\n✅ Found compilation database: {compdb}")

    # Set output directory
    output_dir = Path(args.output_dir).absolute() if args.output_dir else repo_path / "scip_output"
    print(f"Output directory: {output_dir}")

    # Create the indexer
    indexer = SCIPIndexer(project_root=repo_path, output_dir=output_dir, language="cpp")

    # Generate SCIP index
    print("\n" + "=" * 80)
    print("Step 1: Generating SCIP index...")
    print("=" * 80)

    success = indexer.generate_index(
        compdb_path=str(compdb),
        show_compiler_diagnostics=getattr(args, "show_compiler_diagnostics", False),
    )

    if not success:
        print("\n❌ Failed to generate SCIP index")
        return False

    print(f"\n✅ SCIP index generated at: {indexer.index_file}")

    # Decode SCIP index
    print("\n" + "=" * 80)
    print("Step 2: Decoding SCIP index...")
    print("=" * 80)

    success = indexer.decode_index()

    if not success:
        print("\n❌ Failed to decode SCIP index")
        return False

    print(f"\n✅ SCIP index decoded at: {indexer.decoded_file}")

    # Show file sizes
    index_size = indexer.index_file.stat().st_size / (1024 * 1024)
    decoded_size = indexer.decoded_file.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 80)
    print("Output Files:")
    print("=" * 80)
    print(f"Binary SCIP index: {indexer.index_file} ({index_size:.2f} MB)")
    print(f"Decoded SCIP index: {indexer.decoded_file} ({decoded_size:.2f} MB)")
    print("=" * 80)

    # Verify output if it's test_cpp_simple
    if "test_cpp_simple" in str(repo_path):
        verify_scip_output(indexer.decoded_file, "C++")

    print("\n✅ Test completed successfully!")
    return True


def test_cpp_multisweb(args):
    """Test C++ SCIP indexer with Multi-SWE-bench dataset."""
    from codeminer.env import load_filter_swebench_dataset, process_swebench_instance

    args_dict = {
        "dataset": "SWE-bench/SWE-bench_Multilingual",
        "split": "test",
        "filter_instance": ".*",
    }

    dataset_args = argparse.Namespace(**args_dict)
    all_dataset = load_filter_swebench_dataset(args=dataset_args)

    # Filter for C/C++ projects
    cpp_instances = []
    cpp_repo_keywords = ["fmtlib/", "nlohmann/", "jqlang/", "redis/", "valkey-io/", "micropython/"]

    for inst in all_dataset:
        repo = inst["repo"]
        if any(keyword in repo for keyword in cpp_repo_keywords):
            cpp_instances.append(inst)

    if len(cpp_instances) == 0:
        print(f"❌ No C/C++ instances found in dataset")
        return False

    # Limit dataset
    if args.num_instances is not None:
        dataset = cpp_instances[: min(args.num_instances, len(cpp_instances))]
    else:
        dataset = cpp_instances

    print(f"\n✓ Found {len(cpp_instances)} total C/C++ instances, testing {len(dataset)}")

    success_count = 0
    failed_instances = []

    for idx, instance in enumerate(dataset):
        instance_id = instance["instance_id"]
        print(f"\n{'=' * 80}")
        print(f"Processing C++ instance [{idx + 1}/{len(dataset)}]: {instance_id}")
        print(f"{'=' * 80}")

        try:
            repo_path = process_swebench_instance(instance)
            output_path = f"./scip_output/cpp/{instance_id}"

            if not generate_cpp_compilation_database(Path(repo_path)):
                failed_instances.append((instance_id, "Could not generate compilation database"))
                continue

            indexer = SCIPIndexer(repo_path, output_dir=output_path, language="cpp")

            if not indexer.generate_index(show_compiler_diagnostics=False):
                failed_instances.append((instance_id, "Failed to generate index"))
                continue

            if not indexer.decode_index():
                failed_instances.append((instance_id, "Failed to decode index"))
                continue

            print(f"\n✅ Successfully processed {instance_id}")
            success_count += 1

        except Exception as e:
            print(f"\n❌ Error processing {instance_id}: {str(e)}")
            failed_instances.append((instance_id, str(e)))

    print(f"\n{'=' * 80}")
    print(f"Total: {len(dataset)} | Success: {success_count} | Failed: {len(failed_instances)}")
    print(f"{'=' * 80}")

    return success_count == len(dataset)


# ==============================================================================
# Rust Build and Indexing
# ==============================================================================


def build_rust_project(repo_path: Path, clean: bool = False) -> bool:
    """
    Build the Rust project using Cargo.

    Args:
        repo_path: Path to the repository
        clean: Whether to clean before building

    Returns:
        bool: True if build was successful
    """
    if clean:
        print(f"Cleaning build artifacts...")
        result = subprocess.run(
            ["cargo", "clean"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✅ Clean successful")

    print("\n" + "=" * 80)
    print("Building Rust project with Cargo...")
    print("=" * 80)

    result = subprocess.run(
        ["cargo", "build"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ Cargo build failed:")
        print(result.stderr)
        return False

    print("✅ Build successful")
    return True


def test_rust_simple(args):
    """Test Rust SCIP indexer with a local repository."""
    repo_path = Path(args.repo).absolute()

    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        return False

    print("=" * 80)
    print("SCIP Rust Indexer Test - Simple Mode")
    print("=" * 80)
    print(f"Repository path: {repo_path}")
    print("=" * 80)

    # Check for Cargo.toml
    if not (repo_path / "Cargo.toml").exists():
        print("\n❌ Error: No Cargo.toml found!")
        return False

    # Build the project
    print("\n📦 Building Rust project...")
    if not build_rust_project(repo_path, clean=args.clean):
        print("\n❌ Build failed")
        return False

    # Set output directory
    output_dir = Path(args.output_dir).absolute() if args.output_dir else repo_path / "scip_output"
    print(f"\nOutput directory: {output_dir}")

    # Create the indexer
    indexer = SCIPIndexer(project_root=repo_path, output_dir=output_dir, language="rust")

    # Generate SCIP index
    print("\n" + "=" * 80)
    print("Step 1: Generating SCIP index...")
    print("=" * 80)

    success = indexer.generate_index(
        exclude_vendored_libraries=getattr(args, "exclude_vendored", True),
    )

    if not success:
        print("\n❌ Failed to generate SCIP index")
        return False

    print(f"\n✅ SCIP index generated at: {indexer.index_file}")

    # Decode SCIP index
    print("\n" + "=" * 80)
    print("Step 2: Decoding SCIP index...")
    print("=" * 80)

    success = indexer.decode_index()

    if not success:
        print("\n❌ Failed to decode SCIP index")
        return False

    print(f"\n✅ SCIP index decoded at: {indexer.decoded_file}")

    # Show file sizes
    index_size = indexer.index_file.stat().st_size / (1024 * 1024)
    decoded_size = indexer.decoded_file.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 80)
    print("Output Files:")
    print("=" * 80)
    print(f"Binary SCIP index: {indexer.index_file} ({index_size:.2f} MB)")
    print(f"Decoded SCIP index: {indexer.decoded_file} ({decoded_size:.2f} MB)")
    print("=" * 80)

    # Verify output if it's test_rust_simple
    if "test_rust_simple" in str(repo_path):
        verify_scip_output(indexer.decoded_file, "Rust")

    print("\n✅ Test completed successfully!")
    return True


def test_rust_multisweb(args):
    """Test Rust SCIP indexer with Multi-SWE-bench dataset."""
    from codeminer.env import load_filter_swebench_dataset, process_swebench_instance

    args_dict = {
        "dataset": "SWE-bench/SWE-bench_Multilingual",
        "split": "test",
        "filter_instance": ".*",
    }

    dataset_args = argparse.Namespace(**args_dict)
    all_dataset = load_filter_swebench_dataset(args=dataset_args)

    # Filter for Rust projects
    rust_instances = []
    rust_repo_keywords = ["tokio-rs/", "serde-rs/", "clap-rs/", "rayon-rs/", "sharkdp/", "nushell/"]

    for inst in all_dataset:
        repo = inst["repo"]
        if any(keyword in repo for keyword in rust_repo_keywords):
            rust_instances.append(inst)

    if len(rust_instances) == 0:
        print(f"❌ No Rust instances found in dataset")
        return False

    # Limit dataset
    if args.num_instances is not None:
        dataset = rust_instances[: min(args.num_instances, len(rust_instances))]
    else:
        dataset = rust_instances

    print(f"\n✓ Found {len(rust_instances)} total Rust instances, testing {len(dataset)}")

    success_count = 0
    failed_instances = []

    for idx, instance in enumerate(dataset):
        instance_id = instance["instance_id"]
        print(f"\n{'=' * 80}")
        print(f"Processing Rust instance [{idx + 1}/{len(dataset)}]: {instance_id}")
        print(f"{'=' * 80}")

        try:
            repo_path = process_swebench_instance(instance)
            output_path = f"./scip_output/rust/{instance_id}"

            if not (Path(repo_path) / "Cargo.toml").exists():
                failed_instances.append((instance_id, "No Cargo.toml found"))
                continue

            indexer = SCIPIndexer(repo_path, output_dir=output_path, language="rust")

            if not indexer.generate_index(exclude_vendored_libraries=True):
                failed_instances.append((instance_id, "Failed to generate index"))
                continue

            if not indexer.decode_index():
                failed_instances.append((instance_id, "Failed to decode index"))
                continue

            print(f"\n✅ Successfully processed {instance_id}")
            success_count += 1

        except Exception as e:
            print(f"\n❌ Error processing {instance_id}: {str(e)}")
            failed_instances.append((instance_id, str(e)))

    print(f"\n{'=' * 80}")
    print(f"Total: {len(dataset)} | Success: {success_count} | Failed: {len(failed_instances)}")
    print(f"{'=' * 80}")

    return success_count == len(dataset)


# ==============================================================================
# TypeScript Indexing
# ==============================================================================


def test_ts_simple(args):
    """Test TypeScript SCIP indexer with a local repository."""
    repo_path = Path(args.repo).absolute()

    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        return False

    print("=" * 80)
    print("SCIP TypeScript Indexer Test - Simple Mode")
    print("=" * 80)
    print(f"Repository path: {repo_path}")
    print("=" * 80)

    # Install dependencies if package.json exists
    package_json = repo_path / "package.json"
    if package_json.exists():
        print("\n📦 Installing npm dependencies...")
        result = subprocess.run(
            ["npm", "install"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            print("✅ npm install successful")
        else:
            print(f"⚠️  npm install failed: {result.stderr[:200]}")

    # Set output directory
    output_dir = Path(args.output_dir).absolute() if args.output_dir else repo_path / "scip_output"
    print(f"\nOutput directory: {output_dir}")

    # Create the indexer
    indexer = SCIPIndexer(project_root=repo_path, output_dir=output_dir, language="ts")

    # Generate SCIP index
    print("\n" + "=" * 80)
    print("Step 1: Generating SCIP index...")
    print("=" * 80)

    # Check if tsconfig.json exists
    tsconfig = repo_path / "tsconfig.json"
    infer_tsconfig = not tsconfig.exists()

    if infer_tsconfig:
        print("⚠️  No tsconfig.json found, will infer TypeScript configuration")

    success = indexer.generate_index(infer_tsconfig=infer_tsconfig)

    if not success:
        print("\n❌ Failed to generate SCIP index")
        return False

    print(f"\n✅ SCIP index generated at: {indexer.index_file}")

    # Decode SCIP index
    print("\n" + "=" * 80)
    print("Step 2: Decoding SCIP index...")
    print("=" * 80)

    success = indexer.decode_index()

    if not success:
        print("\n❌ Failed to decode SCIP index")
        return False

    print(f"\n✅ SCIP index decoded at: {indexer.decoded_file}")

    # Show file sizes
    index_size = indexer.index_file.stat().st_size / (1024 * 1024)
    decoded_size = indexer.decoded_file.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 80)
    print("Output Files:")
    print("=" * 80)
    print(f"Binary SCIP index: {indexer.index_file} ({index_size:.2f} MB)")
    print(f"Decoded SCIP index: {indexer.decoded_file} ({decoded_size:.2f} MB)")
    print("=" * 80)

    # Verify output if it's test_typescript_simple
    if "test_typescript_simple" in str(repo_path):
        verify_scip_output(indexer.decoded_file, "TypeScript")

    print("\n✅ Test completed successfully!")
    return True


def test_ts_multisweb(args):
    """Test TypeScript SCIP indexer with Multi-SWE-bench dataset."""
    from codeminer.env import load_filter_swebench_dataset, process_swebench_instance

    args_dict = {
        "dataset": "SWE-bench/SWE-bench_Multilingual",
        "split": "test",
        "filter_instance": ".*",
    }

    dataset_args = argparse.Namespace(**args_dict)
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
        return False

    # Limit dataset
    if args.num_instances is not None:
        dataset = ts_instances[: min(args.num_instances, len(ts_instances))]
    else:
        dataset = ts_instances

    print(f"\n✓ Found {len(ts_instances)} total TypeScript instances, testing {len(dataset)}")

    success_count = 0
    failed_instances = []

    for idx, instance in enumerate(dataset):
        instance_id = instance["instance_id"]
        print(f"\n{'=' * 80}")
        print(f"Processing TypeScript instance [{idx + 1}/{len(dataset)}]: {instance_id}")
        print(f"{'=' * 80}")

        try:
            repo_path = process_swebench_instance(instance)
            output_path = f"./scip_output/ts/{instance_id}"

            # Install dependencies if package.json exists
            package_json = Path(repo_path) / "package.json"
            if package_json.exists():
                print("  Installing dependencies...")
                subprocess.run(
                    ["npm", "install"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

            indexer = SCIPIndexer(repo_path, output_dir=output_path, language="ts")

            # Check if tsconfig.json exists
            tsconfig = Path(repo_path) / "tsconfig.json"
            infer_tsconfig = not tsconfig.exists()

            if not indexer.generate_index(infer_tsconfig=infer_tsconfig):
                failed_instances.append((instance_id, "Failed to generate index"))
                continue

            if not indexer.decode_index():
                failed_instances.append((instance_id, "Failed to decode index"))
                continue

            print(f"\n✅ Successfully processed {instance_id}")
            success_count += 1

        except Exception as e:
            print(f"\n❌ Error processing {instance_id}: {str(e)}")
            failed_instances.append((instance_id, str(e)))

    print(f"\n{'=' * 80}")
    print(f"Total: {len(dataset)} | Success: {success_count} | Failed: {len(failed_instances)}")
    print(f"{'=' * 80}")

    return success_count == len(dataset)


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Unified SCIP indexer test for C++, Rust, and TypeScript"
    )

    # Language selection
    parser.add_argument(
        "--lang",
        choices=["cpp", "rust", "ts"],
        required=True,
        help="Language to test (cpp, rust, ts)",
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--repo",
        type=str,
        help="Path to local repository (simple mode)",
    )
    mode_group.add_argument(
        "--multisweb",
        action="store_true",
        help="Test with Multi-SWE-bench dataset",
    )

    # Common arguments
    parser.add_argument(
        "--clean",
        action="store_true",
        help="[Simple mode] Clean build artifacts before building",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="[Simple mode] Output directory for SCIP files",
    )

    # C++ specific arguments
    parser.add_argument(
        "--show-compiler-diagnostics",
        action="store_true",
        help="[C++ only] Show compiler diagnostics during indexing",
    )

    # Rust specific arguments
    parser.add_argument(
        "--exclude-vendored",
        action="store_true",
        default=True,
        help="[Rust only] Exclude vendored libraries from indexing (default: True)",
    )

    # Multi-SWE-bench mode arguments
    parser.add_argument(
        "--num-instances",
        type=int,
        default=None,
        help="[Multi-SWE-bench mode] Number of instances to test",
    )

    args = parser.parse_args()

    # Run test based on language and mode
    if args.lang == "cpp":
        if args.multisweb:
            print("\n=== Running C++ Multi-SWE-bench mode ===\n")
            success = test_cpp_multisweb(args)
        elif args.repo:
            print("\n=== Running C++ Simple mode ===\n")
            success = test_cpp_simple(args)
        else:
            parser.error("Please specify either --repo or --multisweb")

    elif args.lang == "rust":
        if args.multisweb:
            print("\n=== Running Rust Multi-SWE-bench mode ===\n")
            success = test_rust_multisweb(args)
        elif args.repo:
            print("\n=== Running Rust Simple mode ===\n")
            success = test_rust_simple(args)
        else:
            parser.error("Please specify either --repo or --multisweb")

    elif args.lang == "ts":
        if args.multisweb:
            print("\n=== Running TypeScript Multi-SWE-bench mode ===\n")
            success = test_ts_multisweb(args)
        elif args.repo:
            print("\n=== Running TypeScript Simple mode ===\n")
            success = test_ts_simple(args)
        else:
            parser.error("Please specify either --repo or --multisweb")

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
