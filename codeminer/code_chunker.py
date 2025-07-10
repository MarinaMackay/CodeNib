#!/usr/bin/env python3
"""
Legacy wrapper for the refactored code chunking system.
This file maintains backward compatibility with the old API.
"""

# Import from the new modular code chunking system
from .code_chunking import create_chunker


# For backward compatibility, expose the old class name
class CodeChunker:
    """Legacy wrapper for the new modular code chunking system."""

    def __init__(self, language: str = "python"):
        """
        Initialize the code chunker for a specific language.

        Args:
            language: Programming language to parse ('python', 'cpp', 'java', etc.)
        """
        self._chunker = create_chunker(language)

    def chunk_file(self, file_path: str):
        """Chunk a code file into function/class level pieces."""
        return self._chunker.chunk_file(file_path)

    def save_chunks_to_json(self, chunks, output_path: str):
        """Save chunks to a JSON file."""
        return self._chunker.save_chunks_to_json(chunks, output_path)

    def print_chunk_summary(self, chunks):
        """Print a summary of the generated chunks."""
        return self._chunker.print_chunk_summary(chunks)


# Legacy main function for backward compatibility
def main():
    """Main function for command-line usage (legacy)."""
    import argparse
    from pathlib import Path

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

    # Create chunker
    chunker = CodeChunker(language=args.language)

    # Chunk the file
    chunks = chunker.chunk_file(args.file)

    if not chunks:
        print("No chunks generated")
        return

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


if __name__ == "__main__":
    main()
