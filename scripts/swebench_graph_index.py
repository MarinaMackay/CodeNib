"""
This script preprocesses and creates graph indexes for SWE-bench instances.
It generates SCIP-based code graphs for each instance in the dataset.

Usage example:
    python scripts/swebench_graph_index.py --dataset princeton-nlp/SWE-bench_Lite --split test --filter-instance "^(django__django-11099)$" --output-path /tmp/swebench_indexes
"""

import argparse
from pathlib import Path

from codeminer.env.process_swebench_data import (
    load_filter_swebench_dataset,
    process_swebench_instance,
)
from codeminer.log_utils import get_logger
from codeminer.scip_interface.scip_indexer import SCIPIndexer

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess and create graph indexes for SWE-bench instances",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Verified",
        help="SWE-bench dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--filter-instance",
        type=str,
        default=".*",
        help="Regex pattern to filter instances (default: .* processes all instances)",
    )

    # Index selection
    parser.add_argument(
        "--idx-range",
        type=int,
        nargs=2,
        default=None,
        help="Range of instance indices to process (e.g., 0 10 for instances 0-9)",
    )

    # Output configuration
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to store the generated graph indexes (default: ~/.codeminer/)",
    )

    # SCIP indexer configuration
    parser.add_argument(
        "--exclude-patterns",
        type=str,
        nargs="+",
        default=["test*"],
        help="Patterns to exclude from indexing (e.g., test/ docs/)",
    )
    parser.add_argument(
        "--skip-level",
        type=str,
        choices=["graph", "decode", "raw", None],
        default="graph",
        help="Cache/skip level: 'graph' (check graph.pkl), 'decode' (check index.decoded), 'raw' (check index.scip), or None (run full pipeline)",
    )

    # Repository cache
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="~/.codeminer",
        help="Directory to cache repositories",
    )

    return parser.parse_args()


def main():
    """Main function to process SWE-bench instances and create graph indexes."""
    args = parse_args()

    # Load and filter dataset
    logger.info(f"Loading dataset: {args.dataset}, split: {args.split}")
    logger.info(f"Filter pattern: {args.filter_instance}")

    dataset = load_filter_swebench_dataset(args)

    logger.info(f"Loaded {len(dataset)} instances from dataset")

    # Set default output index path if not specified
    if args.output_path is None:
        args.output_path = str(Path.home() / ".codeminer")

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Graph indexes will be stored in: {output_path}")

    # Process each instance
    for idx, instance in enumerate(dataset):
        instance_id = instance["instance_id"]
        repo = instance["repo"]
        base_commit = instance["base_commit"]

        logger.info(f"\n{'='*80}")
        logger.info(f"Processing instance {idx + 1}/{len(dataset)}: {instance_id}")
        logger.info(f"Repository: {repo}")
        logger.info(f"Base commit: {base_commit}")
        logger.info(f"{'='*80}")

        try:
            # Process the instance (clone repo and checkout commit)
            repo_path = process_swebench_instance(instance, cache_dir=args.cache_dir)
            logger.info(f"Repository checked out at: {repo_path}")

            # Create output directory for this instance
            instance_output_dir = output_path / instance_id.replace("/", "__")
            instance_output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize SCIP indexer
            indexer = SCIPIndexer(
                project_root=repo_path,
                output_dir=instance_output_dir,
                exclude_patterns=args.exclude_patterns,
            )

            # Run the indexing pipeline
            logger.info(f"Starting graph indexing for {instance_id}")
            graph = indexer.run_pipeline(
                project_name=instance_id,
                skip_level=args.skip_level,
            )
            indexer.clear_cache(level="graph")

            if graph:
                logger.info(f"  Successfully created graph index for {instance_id}")
                logger.info(f"   Graph saved to: {indexer.graph_file}")
                logger.info(
                    f"   Graph stats: {len(graph.graph.vs)} nodes, {len(graph.graph.es)} edges"
                )
            else:
                logger.error(f"L Failed to create graph index for {instance_id}")

        except Exception as e:
            logger.error(f"L Error processing instance {instance_id}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            continue

    logger.info(f"\n{'='*80}")
    logger.info("Graph indexing complete!")
    logger.info(f"Processed {len(dataset)} instances")
    logger.info(f"Indexes stored in: {output_path}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
