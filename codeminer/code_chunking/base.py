#!/usr/bin/env python3
"""
Base code chunker class with common functionality.
"""

import json
import os
import sys
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, Optional, Tuple

# Import the tree-sitter-language-pack
from tree_sitter_language_pack import get_language, get_parser

from ..log_utils import get_logger

logger = get_logger(__name__)

# Define structures for code chunks
CodeChunk = namedtuple(
    "CodeChunk",
    ["content", "start_line", "end_line", "chunk_type", "name", "file", "node_id"],
)


class BaseCodeChunker(ABC):
    """Base class for language-specific code chunkers."""

    def __init__(
        self,
        language: str,
        max_lines_per_chunk: Optional[int] = None,
        chunk_depth: int = 2,
        enable_max_split: bool = True,
        include_header_epilogue: bool = False,
    ):
        """
        Initialize the code chunker for a specific language.

        Args:
            language: Programming language to parse ('python', 'cpp', 'java', etc.)
            max_lines_per_chunk: Maximum number of lines per emitted chunk. When set,
                large logical chunks (function/class) will be split into
                multiple sequential chunks of at most this many lines. node_id and name remain
                the same across the split pieces. Default: None (no splitting). Set to a number to enable.
            chunk_depth: Depth of AST traversal for chunking:
                1 = Top-level only (classes and top-level functions, no methods)
                2 = Method-level (classes, functions, and methods) [default]
            enable_max_split: Whether to apply max_lines_per_chunk splitting. When False,
                keeps logical units (functions/classes/methods) intact regardless of size.
            include_header_epilogue: Whether to include file header (imports, module docstrings)
                and epilogue (trailing code) in chunks. Default: False (skip them to reduce noise).
        """
        self.language = language
        self.max_lines_per_chunk = max_lines_per_chunk
        self.chunk_depth = chunk_depth
        self.enable_max_split = enable_max_split
        self.include_header_epilogue = include_header_epilogue
        try:
            self.parser = get_parser(language)
            self.tree_sitter_language = get_language(language)
            logger.info(
                f"Successfully loaded {language} language parser from tree-sitter-language-pack"
            )
        except Exception as e:
            logger.error(f"Error loading {language} parser: {e}")
            sys.exit(1)

    def chunk_file(
        self, file_path: str, relative_path: Optional[str] = None
    ) -> List[CodeChunk]:
        """
        Chunk a code file into function/class level pieces.

        Args:
            file_path: Absolute path to the code file to chunk
            relative_path: Relative path for node_id generation

        Returns:
            List of CodeChunk objects representing the chunks
        """
        if not os.path.exists(file_path):
            logger.error(f"Error: File {file_path} not found")
            return []

        # logger.debug(f"Chunking file: {file_path}")

        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            code_content = f.read()

        # Parse the code
        code_bytes = code_content.encode("utf-8")
        tree = self.parser.parse(code_bytes)
        root_node = tree.root_node

        # Split content into lines for easier chunk extraction
        lines = code_content.split("\n")

        # Find all top-level functions and classes
        top_level_nodes = self._find_top_level_definitions(root_node)

        # Use relative_path for node_id generation, fallback to file_path
        path_for_node_id = relative_path if relative_path else file_path

        # Generate chunks
        chunks = self._generate_chunks(
            lines, top_level_nodes, file_path, path_for_node_id
        )

        # logger.debug(f"Generated {len(chunks)} chunks from {file_path}")
        return chunks

    def _split_by_max_lines(
        self,
        lines: List[str],
        start_line: int,
        end_line: int,
        chunk_type: str,
        name: str,
        file_path: str,
        node_id: str,
    ) -> List[CodeChunk]:
        """
        Split a logical chunk into multiple CodeChunk pieces if it exceeds max_lines_per_chunk.
        Keeps node_id and name unchanged across pieces.
        Uses balanced splitting to distribute lines evenly across chunks.
        """
        # If max split is disabled, or no max specified, or chunk already within limit, return single piece
        if (
            not self.enable_max_split
            or not self.max_lines_per_chunk
            or self.max_lines_per_chunk <= 0
        ):
            # Build prefix with node_id and class context for methods
            prefix_lines = [node_id]
            if chunk_type == "method" and "." in name:
                class_name = name.split(".")[0]
                prefix_lines.append(f"class {class_name}:")
            prefix = "\n".join(prefix_lines) + "\n"

            content = prefix + "\n".join(lines[start_line : end_line + 1])
            return [
                CodeChunk(
                    content=content,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type=chunk_type,
                    name=name,
                    file=file_path,
                    node_id=node_id,
                )
            ]

        total_lines = end_line - start_line + 1
        if total_lines <= self.max_lines_per_chunk:
            # Build prefix with node_id and class context for methods
            prefix_lines = [node_id]
            if chunk_type == "method" and "." in name:
                class_name = name.split(".")[0]
                prefix_lines.append(f"class {class_name}:")
            prefix = "\n".join(prefix_lines) + "\n"

            content = prefix + "\n".join(lines[start_line : end_line + 1])
            return [
                CodeChunk(
                    content=content,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type=chunk_type,
                    name=name,
                    file=file_path,
                    node_id=node_id,
                )
            ]

        # Calculate number of chunks needed and balanced chunk sizes
        num_chunks = (
            total_lines + self.max_lines_per_chunk - 1
        ) // self.max_lines_per_chunk
        base_chunk_size = total_lines // num_chunks
        extra_lines = total_lines % num_chunks

        # Create balanced chunks
        pieces: List[CodeChunk] = []
        current_start = start_line

        # Build prefix with node_id and class context for methods
        prefix_lines = [node_id]
        if chunk_type == "method" and "." in name:
            class_name = name.split(".")[0]
            prefix_lines.append(f"class {class_name}:")
        prefix = "\n".join(prefix_lines) + "\n"

        for i in range(num_chunks):
            chunk_size = base_chunk_size + (1 if i < extra_lines else 0)
            current_end = current_start + chunk_size - 1

            piece_content = prefix + "\n".join(lines[current_start : current_end + 1])
            pieces.append(
                CodeChunk(
                    content=piece_content,
                    start_line=current_start,
                    end_line=current_end,
                    chunk_type=chunk_type,
                    name=name,
                    file=file_path,
                    node_id=node_id,
                )
            )
            current_start = current_end + 1

        return pieces

    def _generate_chunks(
        self,
        lines: List[str],
        definitions: List[Tuple],
        file_path: str,
        path_for_node_id: str,
    ) -> List[CodeChunk]:
        """
        Generate code chunks from the file content and AST definitions.

        Args:
            lines: List of code lines
            definitions: List of (node, name, type) tuples
            file_path: Path to the source file
            path_for_node_id: Path to use for node_id generation

        Returns:
            List of CodeChunk objects
        """
        chunks: List[CodeChunk] = []
        current_line = 0

        for i, (node, name, def_type) in enumerate(definitions):
            start_line = node.start_point[0]  # 0-based
            end_line = node.end_point[0]  # 0-based

            # Create chunk for code before this definition (only for the first one)
            if i == 0 and start_line > current_line and self.include_header_epilogue:
                header_content_lines = lines[current_line:start_line]
                if "\n".join(header_content_lines).strip():  # Only add if not empty
                    # node_id for headers is file path (no symbol)
                    chunks.extend(
                        self._split_by_max_lines(
                            lines=lines,
                            start_line=current_line,
                            end_line=start_line - 1,
                            chunk_type="header",
                            name="header",
                            file_path=file_path,
                            node_id=path_for_node_id,
                        )
                    )

            # Create chunk for the function/class definition
            # Generate node_id in graph format: file_path:symbol_name
            node_id = self._generate_node_id(path_for_node_id, name, def_type)
            chunks.extend(
                self._split_by_max_lines(
                    lines=lines,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type=def_type,
                    name=name,
                    file_path=file_path,
                    node_id=node_id,
                )
            )

            current_line = end_line + 1

        # Handle any remaining code after the last definition
        if current_line < len(lines) and self.include_header_epilogue:
            remaining_content_lines = lines[current_line:]
            if "\n".join(remaining_content_lines).strip():  # Only add if not empty
                chunks.extend(
                    self._split_by_max_lines(
                        lines=lines,
                        start_line=current_line,
                        end_line=len(lines) - 1,
                        chunk_type="epilogue",
                        name="epilogue",
                        file_path=file_path,
                        node_id=path_for_node_id,
                    )
                )

        return chunks

    def _generate_node_id(
        self, file_path: str, symbol_name: str, symbol_type: str
    ) -> str:
        """
        Generate node_id in the same format as code graph.

        Args:
            file_path: Path to the file (should be relative path)
            symbol_name: Name of the symbol (function/class/method name)
            symbol_type: Type of the symbol ("function", "method", or "class")

        Returns:
            Node ID in format: file_path:symbol_name
        """
        # For functions and methods, add parentheses to match graph format
        if symbol_type in ("function", "method"):
            formatted_name = f"{symbol_name}()"
        else:
            formatted_name = symbol_name

        return f"{file_path}:{formatted_name}"

    @abstractmethod
    def _find_top_level_definitions(self, root_node) -> List[Tuple]:
        """
        Find all top-level function and class definitions.

        Args:
            root_node: Root node of the AST

        Returns:
            List of tuples (node, name, type) for each top-level definition
        """
        pass

    @abstractmethod
    def _extract_function_name(self, node) -> Optional[str]:
        """
        Extract function name from function definition node.

        Args:
            node: AST node representing a function definition

        Returns:
            Function name or None if extraction failed
        """
        pass

    @abstractmethod
    def _extract_class_name(self, node) -> Optional[str]:
        """
        Extract class name from class definition node.

        Args:
            node: AST node representing a class definition

        Returns:
            Class name or None if extraction failed
        """
        pass

    def _find_nodes_by_type(self, root_node, node_type: str):
        """Find all nodes with the given type in the tree."""
        nodes = []

        def traverse(node):
            if node.type == node_type:
                nodes.append(node)
            for child in node.children:
                traverse(child)

        traverse(root_node)
        return nodes

    def save_chunks_to_json(self, chunks: List[CodeChunk], output_path: str):
        """
        Save chunks to a JSON file.

        Args:
            chunks: List of CodeChunk objects
            output_path: Path to save the JSON file
        """
        # Convert chunks to dictionaries for JSON serialization
        chunk_dicts = [
            {
                "content": chunk.content,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chunk_type": chunk.chunk_type,
                "name": chunk.name,
                "file": chunk.file,
                "node_id": chunk.node_id,
            }
            for chunk in chunks
        ]

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunk_dicts, f, indent=2, ensure_ascii=False)
            logger.info(f"Chunks saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving chunks to file: {e}")

    def print_chunk_summary(self, chunks: List[CodeChunk]):
        """Print a summary of the generated chunks."""
        logger.info(f"\n=== Chunk Summary ===")
        logger.info(f"Total chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks, 1):
            logger.info(
                f"Chunk {i}: {chunk.chunk_type} '{chunk.name}' "
                f"(lines {chunk.start_line}-{chunk.end_line}, "
                f"{len(chunk.content.split(chr(10)))} lines)"
            )
