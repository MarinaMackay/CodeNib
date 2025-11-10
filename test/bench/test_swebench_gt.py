#!/usr/bin/env python3
"""
Test script for the ground truth locator with unified diff generation.
Generates annotated diff reports showing detected changes.

Usage:
    python test/bench/test_swebench_gt.py --limit 1
    python test/bench/test_swebench_gt.py --filter "astropy__astropy-.*" --limit 5
"""

import argparse
import difflib
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from codeminer.log_utils import get_logger  # noqa: E402
from scripts.swebench_gt_locate import GTLocator, load_swebench_verified  # noqa: E402

logger = get_logger(__name__)


def generate_unified_diff_with_full_context(
    content_before: str, content_after: str, filename: str
) -> list:
    """
    Generate GitHub-style unified diff showing the complete file.

    Returns:
        List of lines in unified diff format
    """
    lines_before = content_before.splitlines(keepends=True)
    lines_after = content_after.splitlines(keepends=True)

    # Use difflib to generate unified diff with large context to show all lines
    diff = difflib.unified_diff(
        lines_before,
        lines_after,
        fromfile=f"{filename} (BEFORE)",
        tofile=f"{filename} (AFTER)",
        lineterm="",
        n=999999,  # Large context value to display all lines
    )

    return list(diff)


def create_annotated_diff_report(
    instance, result: dict, locator: GTLocator, output_dir: str
):
    """
    Create annotated diff report with detection results.

    Outputs a single consolidated file containing:
    1. Detection result summary
    2. Complete unified diff for each file
    3. Annotations for detected symbols
    """
    instance_id = result["instance_id"]
    repo = result["repo"]
    base_commit = result["base_commit"]

    # Create instance-specific directory
    instance_dir = os.path.join(output_dir, instance_id)
    os.makedirs(instance_dir, exist_ok=True)

    # Get repository directory
    repo_dir_name = repo.replace("/", "_")
    repo_dir = os.path.join(locator.work_dir, repo_dir_name)

    # Prepare output file
    output_file = os.path.join(instance_dir, "report.txt")

    with open(output_file, "w", encoding="utf-8") as out:
        # Header
        out.write("=" * 100 + "\n")
        out.write(f"Instance: {instance_id}\n")
        out.write(f"Repo: {repo}\n")
        out.write(f"Commit: {base_commit}\n")
        out.write("=" * 100 + "\n\n")

        # Detection summary
        out.write("TARGET FILES:\n")
        for file_path in result["target_files"]:
            out.write(f"  {file_path}\n")
        out.write("\n")

        out.write("MODIFIED:\n")
        if result["symbols_modified"]:
            for symbol in result["symbols_modified"]:
                out.write(f"  {symbol}\n")
        else:
            out.write("  (none)\n")

        out.write(f"\nADDED:\n")
        if result["symbols_added"]:
            for symbol in result["symbols_added"]:
                out.write(f"  {symbol}\n")
        else:
            out.write("  (none)\n")

        out.write(f"\nDELETED:\n")
        if result["symbols_deleted"]:
            for symbol in result["symbols_deleted"]:
                out.write(f"  {symbol}\n")
        else:
            out.write("  (none)\n")

        # File diffs
        python_files = [f for f in result["target_files"] if f.endswith(".py")]

        for file_path in python_files:
            full_path = os.path.join(repo_dir, file_path)

            out.write("\n" + "=" * 100 + "\n")
            out.write(f"FILE: {file_path}\n")
            out.write("=" * 100 + "\n\n")

            # List detected symbols in this file
            file_modified = [
                s for s in result["symbols_modified"] if s.startswith(file_path + ":")
            ]
            file_added = [
                s for s in result["symbols_added"] if s.startswith(file_path + ":")
            ]
            file_deleted = [
                s for s in result["symbols_deleted"] if s.startswith(file_path + ":")
            ]

            # Print symbols at the beginning
            out.write("MODIFIED:\n")
            if file_modified:
                for symbol in file_modified:
                    out.write(f"  {symbol.split(':', 1)[1]}\n")
            else:
                out.write("  (none)\n")

            out.write("\nADDED:\n")
            if file_added:
                for symbol in file_added:
                    out.write(f"  {symbol.split(':', 1)[1]}\n")
            else:
                out.write("  (none)\n")

            out.write("\nDELETED:\n")
            if file_deleted:
                for symbol in file_deleted:
                    out.write(f"  {symbol.split(':', 1)[1]}\n")
            else:
                out.write("  (none)\n")

            out.write("\n")

            # Get before content
            locator.checkout_commit(repo_dir, base_commit)
            content_before = ""
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    content_before = f.read()
            else:
                out.write("(File did not exist)\n\n")

            # Apply patch and get after content
            locator.apply_patch(repo_dir, instance["patch"])
            content_after = ""
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    content_after = f.read()
            else:
                out.write("(File deleted)\n\n")
                continue

            # Generate unified diff
            if content_before or content_after:
                diff_lines = generate_unified_diff_with_full_context(
                    content_before, content_after, file_path
                )

                for line in diff_lines:
                    out.write(line + "\n")

                out.write("\n")

    return output_file


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Ground truth locator test with unified diff generation"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="astropy__astropy-.*",
        help="Instance ID filter regex (default: astropy__astropy-.*)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of instances to process (default: 1)",
    )
    parser.add_argument(
        "--work-dir", type=str, default=None, help="Working directory for repositories"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test/bench/results",
        help="Output directory (default: test/bench/results)",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Load dataset
    print(
        f"Loading SWE-bench Verified dataset (filter: {args.filter}, limit: {args.limit})..."
    )
    dataset = load_swebench_verified(
        split="test", filter_pattern=args.filter, limit=args.limit
    )

    print(f"Processing {len(dataset)} instance(s)\n")

    # Initialize locator
    locator = GTLocator(work_dir=args.work_dir)

    # Process each instance
    for i, instance in enumerate(dataset):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(dataset)}] {instance['instance_id']}")
        print("=" * 80)

        try:
            # Analyze instance
            result = locator.analyze_instance(instance)

            if result.get("error"):
                print(f"ERROR: {result['error']}")
                continue

            # Create diff report
            output_file = create_annotated_diff_report(
                instance, result, locator, args.output
            )

            print(f"\nReport saved: {output_file}")

        except Exception as e:
            print(f"EXCEPTION: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"Results saved to: {args.output}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
