#!/usr/bin/env python3
"""
This script builds and caches embedding indices for all SWE-bench Lite instances.
Each instance's embedding will be stored in /mnt/data/codeminer/{instance_id}/

Usage:
    # Build embeddings for all SWE-bench Lite instances
    python examples/build_embeddings.py
    
    # Build with custom embedding model
    python examples/build_embeddings.py --embedding-model nomic-ai/CodeRankEmbed
    
    # Build with filter (for testing)
    python examples/build_embeddings.py --filter-instance "^(astropy__astropy-13579)$"
    
    # Force rebuild even if embeddings already exist
    python examples/build_embeddings.py --force-rebuild
"""

import argparse
import sys
from pathlib import Path

from codeminer.code_chunker import CodeChunker
from codeminer.embedding import CodeVectorStore
from codeminer.env.process_swebench_data import (
    load_filter_swebench_dataset,
    process_swebench_instance,
)
from codeminer.log_utils import get_logger
from codeminer.profiler import Profiler

logger = get_logger(__name__)
profiler = Profiler("build_embeddings")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build embedding indices for SWE-bench Lite instances",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="Dataset name",
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
    
    # Embedding configuration
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nomic-ai/CodeRankEmbed",
        help="Embedding model name for dense retrieval",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default="huggingface",
        choices=["openai", "huggingface"],
        help="Embedding provider",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=768,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code for embedding model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for embedding encoding",
    )
    
    # Repository processing configuration
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["python"],
        help="Programming languages to process",
    )
    parser.add_argument(
        "--max-lines-per-chunk",
        type=int,
        default=300,
        help="Maximum lines per code chunk",
    )
    
    # Storage configuration
    parser.add_argument(
        "--storage-dir",
        type=str,
        default="/mnt/data/codeminer",
        help="Base directory to store embeddings",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        default=False,
        help="Force rebuild embeddings even if they already exist",
    )
    
    # Profiling configuration
    parser.add_argument(
        "--enable-profiler",
        action="store_true",
        default=False,
        help="Enable performance profiling for embedding build times",
    )
    return parser.parse_args()


def build_embeddings(args):
    """Build embedding indices for all SWE-bench Lite instances."""
    
    # Enable/disable profiler based on args
    profiler.enabled = args.enable_profiler
    
    # Prepare dataset args
    dataset_args = argparse.Namespace(
        dataset=args.dataset,
        split=args.split,
        filter_instance=args.filter_instance,
    )
    
    # Load dataset
    dataset_instances = load_filter_swebench_dataset(args=dataset_args)
    
    if len(dataset_instances) == 0:
        raise ValueError(f"No instances found in {args.dataset}")
    
    logger.info(f"Loaded {len(dataset_instances)} instance(s)")
    logger.info(f"Embeddings will be stored in: {args.storage_dir}")
    
    # Process each instance
    for idx, instance in enumerate(dataset_instances):
        instance_id = instance["instance_id"]
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing [{idx+1}/{len(dataset_instances)}]: {instance_id}")
        logger.info(f"{'='*80}")
        
        try:
            # Process instance to get repo path
            repo_path = process_swebench_instance(instance)
            
            # Convert instance_id to directory name (replace / with __)
            instance_dir_name = instance_id.replace("/", "__")
            
            # Set final directory for this instance
            instance_final_dir = Path(args.storage_dir) / instance_dir_name
            instance_final_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Repository path: {repo_path}")
            logger.info(f"Target directory: {instance_final_dir}")
            
            # Check if embedding already exists
            if (instance_final_dir / "config.json").exists() and not args.force_rebuild:
                logger.info(f"✓ Embedding already exists at {instance_final_dir}, skipping...")
                continue
            elif (instance_final_dir / "config.json").exists() and args.force_rebuild:
                logger.info(f"⚠ Embedding already exists but force-rebuild is enabled, rebuilding...")
            
            # Step 1: Chunk the repository code
            logger.info("Chunking repository code...")
            code_chunker = CodeChunker(
                language=args.languages[0],
                max_lines_per_chunk=args.max_lines_per_chunk,
            )
            chunks = code_chunker.chunk_repository(
                repo_path=repo_path,
                languages=args.languages,
            )
            
            if not chunks:
                logger.warning(f"No code chunks generated from repository, skipping...")
                continue
            
            logger.info(f"Generated {len(chunks)} code chunks")
            
            # Step 2: Build vector store (with profiling)
            logger.info("Building vector store...")
            with profiler.section(instance_id):
                # Prepare embedding kwargs
                embedding_kwargs = {}
                if args.trust_remote_code:
                    embedding_kwargs["model_kwargs"] = {"trust_remote_code": True}
                if args.batch_size:
                    embedding_kwargs["encode_kwargs"] = {"batch_size": args.batch_size}
                
                vector_store = CodeVectorStore(
                    embedding_model=args.embedding_model,
                    embedding_provider=args.embedding_provider,
                    dimension=args.embedding_dimension,
                    store_path=str(instance_final_dir),
                    **embedding_kwargs,
                )
                
                # Add chunks to vector store
                chunks_for_indexing = [chunk._asdict() for chunk in chunks]
                vector_store.add_code_chunks(chunks_for_indexing)
            
            # Step 3: Save vector store
            logger.info("Saving vector store...")
            vector_store.save(str(instance_final_dir))
            
            logger.info(f"✓ Successfully built embedding for {instance_id}")
            logger.info(f"  - Total chunks: {len(vector_store.documents)}")
            logger.info(f"  - Saved to: {instance_final_dir}")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {instance_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    logger.info(f"Completed processing {len(dataset_instances)} instance(s)")
    
    # Output profiling report if enabled
    if args.enable_profiler:
        logger.info("Embedding Build Performance Report")
        profiler.report()


def main():
    """Main entry point."""
    args = parse_args()
    
    build_embeddings(args)


if __name__ == "__main__":
    main()

