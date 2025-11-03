#!/usr/bin/env python3
"""
Test script for the ground truth locator.
Run a quick test on a single instance to verify functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.swebench_gt_locate import load_swebench_verified, GTLocator
from codeminer.log_utils import get_logger

logger = get_logger(__name__)


def test_single_instance():
    """Test the locator on a single instance."""
    logger.info("Loading SWE-bench Verified dataset...")
    # Load only the first instance for testing
    dataset = load_swebench_verified(split="test", limit=2)

    if len(dataset) == 0:
        logger.error("No instances found in dataset!")
        return

    # Get first instance
    instance = dataset[0]
    logger.info(f"Testing with instance: {instance['instance_id']}")
    logger.info(f"Repository: {instance['repo']}")
    logger.info(f"Base commit: {instance['base_commit']}")

    # Initialize locator
    locator = GTLocator()

    try:
        # Analyze
        result = locator.analyze_instance(instance)

        # Print results
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS RESULTS")
        logger.info("="*60)
        logger.info(f"Instance ID: {result['instance_id']}")
        logger.info(f"Repository: {result['repo']}")
        logger.info(f"Base Commit: {result['base_commit']}")
        logger.info(f"\nTarget Files ({len(result['target_files'])}):")
        for f in result['target_files']:
            logger.info(f"  - {f}")

        logger.info(f"\nModified Symbols ({len(result['symbols_modified'])}):")
        for f in result['symbols_modified']:
            logger.info(f"  - {f}")

        logger.info(f"\nAdded Symbols ({len(result['symbols_added'])}):")
        for f in result['symbols_added']:
            logger.info(f"  - {f}")

        logger.info(f"\nDeleted Symbols ({len(result['symbols_deleted'])}):")
        for f in result['symbols_deleted']:
            logger.info(f"  - {f}")

        if result.get('error'):
            logger.error(f"\nError: {result['error']}")

        logger.info("="*60)

        return result

    finally:
        # Cleanup
        locator.cleanup()


if __name__ == "__main__":
    test_single_instance()
