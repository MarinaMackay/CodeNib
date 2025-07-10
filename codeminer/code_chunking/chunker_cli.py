#!/usr/bin/env python3
"""
Command-line interface for the code chunking system.
"""

import argparse
from pathlib import Path

from codeminer.code_chunking import create_chunker


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Chunk code files using AST analysis")
    parser.add_argument("file", help="Path to the code file to chunk")
    parser.add_argument(
        "--language",
        "-l",
        default="python",
        help="Programming language (default: python)",
    )
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument(
        "--summary", "-s", action="store_true", help="Print chunk summary"
    )

    args = parser.parse_args()

    try:
        # Create language-specific chunker
        chunker = create_chunker(args.language)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Chunk the file
    chunks = chunker.chunk_file(args.file)

    if not chunks:
        print("No chunks generated")
        return 1

    # Print summary if requested
    if args.summary:
        chunker.print_chunk_summary(chunks)

    # Save to JSON if output path provided
    if args.output:
        chunker.save_chunks_to_json(chunks, args.output)
    else:
        # Default output path
        file_path = Path(args.file)
        default_output = file_path.parent / f"{file_path.stem}_chunks.json"
        chunker.save_chunks_to_json(chunks, str(default_output))

    return 0


if __name__ == "__main__":
    exit(main())
