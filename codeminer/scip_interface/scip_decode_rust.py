#!/usr/bin/env python3
"""
SCIP decoder for Rust projects.

Decodes SCIP index files into CodeGraph format, focusing on:
- Structs (treated as classes)
- Impl blocks and methods
- Free functions
- Traits (treated as classes)
"""
import re
from pathlib import Path

from ..code_chunking import RustCodeChunker
from ..graph.code_graph import CodeGraph
from ..log_utils import get_logger, register_scip_logger
from ..types import (
    EDGE_TYPE_CONTAIN,
    EDGE_TYPE_REFERENCE,
    NODE_TYPE_CLASS,
    NODE_TYPE_FUNCTION,
    NODE_TYPE_METHOD,
    ROOT_NODE,
)


class SCIPRustGraphDecoder:
    """
    Decoder for Rust SCIP indices.

    Parses decoded SCIP index files and builds a CodeGraph representing:
    - Structs as classes
    - Traits as classes
    - Impl blocks and their methods
    - Free functions
    - References between symbols
    """

    def __init__(self, index_file_path, project_root=None):
        """
        Initialize the Rust SCIP decoder.

        Args:
            index_file_path: Path to the decoded SCIP index file
            project_root: Root directory of the Rust project
        """
        self.index_file_path = index_file_path
        self.project_root = Path(project_root) if project_root else None
        self.code_graph = CodeGraph(project_root)
        self.indexed_directories = set()
        self.logger = get_logger(__name__)
        # Register this module for SCIP debug logging
        register_scip_logger(__name__)

        # Initialize code chunker for finding definitions
        self.code_chunker = RustCodeChunker() if project_root else None

        # Cache for file contents to avoid repeated reads
        self.file_contents_cache = {}

        # Cache for file chunks to avoid repeated chunking
        self.file_chunks_cache = {}

    def decode(self):
        """
        Decode the SCIP index and build the CodeGraph.

        Returns:
            CodeGraph: The constructed code graph
        """
        self.logger.info(f"Starting SCIP Rust decode from {self.index_file_path}")
        try:
            with open(self.index_file_path, "r") as f:
                content = f.read()
        except Exception as e:
            self.logger.error(f"Error reading SCIP index file: {e}")
            raise

        # Parse documents
        document_blocks = re.findall(
            r"documents\s*{(.*?)(?=documents\s*{|$)", content, re.DOTALL
        )

        # Add the root node to the graph
        self.code_graph.add_root_node(ROOT_NODE)

        # Process all documents
        for document in document_blocks:
            self._process_document(document)

        return self.code_graph

    def _process_document(self, document_text):
        """
        Process a single document block from the SCIP index.

        Args:
            document_text: Text content of the document block
        """
        # Extract file path
        file_match = re.search(r'relative_path:\s*"([^"]+)"', document_text)
        if not file_match:
            return

        file_path = file_match.group(1)

        # Only process Rust files
        if not file_path.endswith(".rs"):
            return

        # Add directory hierarchy
        dir_path = Path(file_path).parent
        while dir_path != dir_path.parent:  # Stop at the root directory
            dir_path_str = str(dir_path)
            if dir_path_str not in self.indexed_directories:
                self.code_graph.add_directory_node(dir_path_str)
                self.indexed_directories.add(dir_path_str)
                # Add containment edge from parent directory
                self.code_graph._add_edge(
                    str(dir_path.parent), dir_path_str, EDGE_TYPE_CONTAIN
                )
            dir_path = dir_path.parent

        # Add file node
        self.code_graph.add_file_node(file_path)

        # Add file containment edge
        self.code_graph._add_edge(
            str(Path(file_path).parent), file_path, EDGE_TYPE_CONTAIN
        )

        # Process occurrences
        occurrences = re.findall(r"occurrences\s*{(.*?)}", document_text, re.DOTALL)
        for occurrence in occurrences:
            self._process_occurrence(occurrence, file_path)

    def _process_occurrence(self, occurrence_text, file_path):
        """
        Process a single occurrence from the SCIP index.

        Args:
            occurrence_text: Text content of the occurrence
            file_path: Path to the file containing this occurrence
        """
        # Extract ranges
        ranges = re.findall(r"range:\s*(\d+)", occurrence_text)
        if len(ranges) < 3:
            return

        line = int(ranges[0])
        col_start = int(ranges[1])
        col_end = int(ranges[2])

        # Extract symbol
        symbol_match = re.search(r'symbol:\s*"([^"]+)"', occurrence_text)
        if not symbol_match:
            return

        symbol = symbol_match.group(1)

        # Skip stdlib/builtin symbols
        if any(
            lib in symbol
            for lib in [
                " rust-analyzer ",
                "rust-lang/",
                "core/",
                "std/",
                "alloc/",
                "test",
            ]
        ):
            # Allow project symbols but skip standard library
            if not any(proj in symbol for proj in ["cargo ", "crate "]):
                return

        # Skip local symbols
        if "local " in symbol:
            return

        # Extract symbol_roles
        symbol_roles_match = re.search(r"symbol_roles:\s*(\d+)", occurrence_text)
        if symbol_roles_match:
            symbol_roles = int(symbol_roles_match.group(1))
        else:
            symbol_roles = 0  # Reference

        # Extract enclosing range if available
        enclosing_ranges = re.findall(r"enclosing_range:\s*(\d+)", occurrence_text)

        # Process the symbol
        self._process_symbol(
            symbol, file_path, line, col_start, col_end, symbol_roles, enclosing_ranges
        )

    def _extract_hash(self, symbol_part):
        """
        Extract hash from SCIP symbol part (Rust rarely has hashes, but handle if present).

        Args:
            symbol_part: Symbol part like "add(hash)" (rare in Rust)

        Returns:
            Full hash string (16 chars) or None if no hash found
        """
        hash_match = re.search(r"\(([0-9a-f]{8,16})\)", symbol_part)
        if hash_match:
            full_hash = hash_match.group(1)
            return full_hash  # Return full hash to avoid collisions
        return None

    def _remove_hash(self, symbol_part):
        """
        Remove hash from symbol part for display/matching purposes.

        Args:
            symbol_part: Symbol with hash (rare in Rust)

        Returns:
            Symbol without hash
        """
        return re.sub(r"\([0-9a-f]+\)", "", symbol_part)

    def _get_display_name(self, unified_symbol):
        """
        Generate human-readable display name from unified symbol.

        Args:
            unified_symbol: Unified symbol like "myapp/shapes/Rectangle#area" or "math_utils/add"

        Returns:
            Display name like "shapes::Rectangle.area" or "math_utils::add"
        """
        # Remove hash for display
        symbol_no_hash = self._remove_hash(unified_symbol)

        # Convert SCIP separators to Rust style
        if "#" in symbol_no_hash:
            # Method: "shapes/Rectangle#area" -> "shapes::Rectangle.area"
            type_path, member_name = symbol_no_hash.split("#", 1)
            type_path = type_path.replace("/", "::")
            if member_name:
                # Remove trailing () from member if present
                member_name = member_name.rstrip("()")
                return f"{type_path}.{member_name}()"
            else:
                return type_path
        elif "/" in symbol_no_hash:
            # Module/function: "math_utils/add" -> "math_utils::add"
            return symbol_no_hash.replace("/", "::")
        else:
            return symbol_no_hash

    def _unify_symbol_name(self, symbol, file_path):
        """
        Unify Rust symbol names, preserving SCIP format with hash for uniqueness.

        Examples:
        - rust-analyzer cargo test_rust_simple 0.1.0 math_utils/add().
          -> math_utils/add
        - rust-analyzer cargo test_rust_simple 0.1.0 shapes/Rectangle#
          -> shapes/Rectangle#
        - rust-analyzer cargo test_rust_simple 0.1.0 shapes/Shape#area().
          -> shapes/Shape#area

        Args:
            symbol: Original symbol name from SCIP
            file_path: File path where the symbol was found (stored as node attribute)

        Returns:
            Unified symbol name in SCIP format (module/Type#method or module/function)
        """
        # Extract the actual symbol part (after version or URL)
        parts = symbol.split(" ")
        if len(parts) < 4:
            return None

        symbol_part = parts[-1].rstrip(".")

        # Extract hash if present (rare in Rust but handle it)
        hash_part = self._extract_hash(symbol_part)

        # Remove hash temporarily for processing
        symbol_no_hash = self._remove_hash(symbol_part)

        # Clean up trailing markers
        symbol_no_hash = symbol_no_hash.rstrip("#/()")

        # Build unified symbol keeping SCIP format
        # Rust SCIP format: crate/module/Type#method or crate/module/function
        if hash_part:
            # Rare case: has hash
            unified = f"{symbol_no_hash}({hash_part})"
        else:
            # Normal case: no hash, just clean symbol
            unified = symbol_no_hash

        return unified

    def _classify_symbol_type(self, unified_symbol, original_symbol):
        """
        Classify the symbol type based on Rust conventions.

        Args:
            unified_symbol: Unified symbol name in SCIP format (like "shapes/Rectangle#area")
            original_symbol: Original symbol from SCIP

        Returns:
            Symbol type constant (NODE_TYPE_CLASS, NODE_TYPE_METHOD, NODE_TYPE_FUNCTION)
        """
        # Remove hash for classification
        symbol_no_hash = self._remove_hash(unified_symbol)

        # Method: has # with something after it
        if "#" in symbol_no_hash:
            parts = symbol_no_hash.split("#")
            if len(parts) > 1 and parts[1]:
                # Has member after #
                return NODE_TYPE_METHOD
            else:
                # Just type (ends with #)
                return NODE_TYPE_CLASS

        # Check last component to determine type
        last_component = symbol_no_hash.split("/")[-1]

        # Struct/Trait: starts with uppercase
        if last_component and last_component[0].isupper():
            return NODE_TYPE_CLASS

        # Default to function
        return NODE_TYPE_FUNCTION

    def _get_file_content(self, file_path):
        """
        Get the content of a file, using cache if available.

        Args:
            file_path: Relative path to the file

        Returns:
            File content as string, or None if file not found
        """
        if file_path in self.file_contents_cache:
            return self.file_contents_cache[file_path]

        if not self.project_root:
            return None

        full_path = self.project_root / file_path
        if not full_path.exists():
            return None

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.file_contents_cache[file_path] = content
                return content
        except Exception as e:
            self.logger.warning(f"Error reading file {full_path}: {e}")
            return None

    def _get_file_chunks(self, file_path):
        """
        Get chunks for a file, using cache if available.

        Args:
            file_path: Path to the file

        Returns:
            List of chunks or None if not found
        """
        # Check cache first
        if file_path in self.file_chunks_cache:
            return self.file_chunks_cache[file_path]

        if not self.code_chunker or not self.project_root:
            return None

        full_path = self.project_root / file_path
        if not full_path.exists():
            return None

        try:
            # Parse the file and cache the result
            chunks = self.code_chunker.chunk_file(str(full_path))
            self.file_chunks_cache[file_path] = chunks
            return chunks
        except Exception as e:
            self.logger.debug(f"Error chunking file {file_path}: {e}")
            return None

    def _find_definition_range(self, file_path, symbol_name, line):
        """
        Find the definition range for a symbol using the Rust code chunker.

        Args:
            file_path: Path to the file containing the symbol
            symbol_name: Name of the symbol in SCIP format (like "shapes/Rectangle#area")
            line: Line number where the symbol is defined

        Returns:
            Tuple of (start_line, end_line) or None if not found
        """
        # Get chunks from cache
        chunks = self._get_file_chunks(file_path)
        if not chunks:
            return None

        # Remove hash for matching (chunker doesn't include hash)
        symbol_no_hash = self._remove_hash(symbol_name)

        # Convert to display format for matching
        # SCIP format: "shapes/Rectangle#area" -> "shapes::Rectangle.area()"
        display_name = self._get_display_name(symbol_no_hash)

        # Extract simple name and possible variations
        if "." in display_name and "()" in display_name:
            # Method: "shapes::Rectangle.area()" -> try "area", "Rectangle::area", etc.
            parts = display_name.split(".")
            type_path = parts[0]  # e.g., "shapes::Rectangle"
            method_name = parts[-1].rstrip("()")  # e.g., "area"

            # Get simple type name (last component)
            type_name = type_path.split("::")[-1]  # e.g., "Rectangle"

            possible_names = [
                method_name,  # Simple name
                f"{type_name}::{method_name}",  # Type::method
                display_name.rstrip("()"),  # Full path without ()
                f"{type_path}::{method_name}",  # Full qualified with ::
            ]
        elif "::" in display_name:
            # Module function: "math_utils::add" -> try "add", partial paths, full path
            components = display_name.rstrip("()").split("::")
            simple_name = components[-1]

            # Generate possible names with different module depths
            possible_names = [simple_name, display_name.rstrip("()")]
            # Add intermediate qualified names
            for i in range(len(components) - 1, 0, -1):
                possible_names.append("::".join(components[i:]))
        else:
            # Simple name
            possible_names = [display_name.rstrip("()")]

        # Look for the symbol in chunks
        for chunk in chunks:
            chunk_name = getattr(chunk, "name", "")
            if not chunk_name:
                continue

            # Try exact matches first
            for name in possible_names:
                if chunk_name == name or chunk_name == f"{name}()":
                    start_line = getattr(chunk, "start_line", 0)
                    end_line = getattr(chunk, "end_line", start_line)
                    return (start_line, end_line)

                # Try suffix matches
                if chunk_name.endswith(f"::{name}") or chunk_name.endswith(f".{name}"):
                    start_line = getattr(chunk, "start_line", 0)
                    end_line = getattr(chunk, "end_line", start_line)
                    return (start_line, end_line)

        return None

    def _process_symbol(
        self, symbol, file_path, line, col_start, col_end, symbol_roles, enclosing_ranges
    ):
        """
        Process a symbol occurrence and add it to the graph.

        Args:
            symbol: Symbol name from SCIP
            file_path: File containing the symbol
            line: Line number
            col_start: Column start
            col_end: Column end
            symbol_roles: Symbol roles (1=definition, 0=reference)
            enclosing_ranges: Enclosing scope ranges
        """
        self.logger.scip_debug(
            f"Processing Rust symbol: {symbol} at {file_path}:{line}, roles: {symbol_roles}"
        )

        # Unify symbol name
        unified_symbol = self._unify_symbol_name(symbol, file_path)
        if not unified_symbol:
            return

        # Classify symbol type
        symbol_type = self._classify_symbol_type(unified_symbol, symbol)

        # Exit scopes based on current line
        try:
            self.code_graph.exit_scopes_by_line(line)
        except Exception as e:
            self.logger.error(f"Error exiting scopes at line {line}: {e}")
            raise

        # Extract symbol name for definition range lookup
        symbol_name = unified_symbol

        # Handle definition (check if definition bit is set)
        is_definition = symbol_roles & 1  # Bitwise AND to check definition bit
        if is_definition:
            # Try to find definition range using code chunker
            definition_range = None
            if symbol_type in [NODE_TYPE_CLASS, NODE_TYPE_METHOD, NODE_TYPE_FUNCTION]:
                # Extract just the function/method/class name
                if "." in symbol_name:
                    # Method: "StructName.method()" -> "method"
                    simple_name = symbol_name.split(".")[-1].rstrip("()")
                else:
                    # Function or class: "function()" -> "function"
                    simple_name = symbol_name.rstrip("()")

                definition_range = self._find_definition_range(file_path, simple_name, line)

            if definition_range:
                start_line, end_line = definition_range
                self.code_graph.add_symbol_node(
                    unified_symbol, line, start_line, end_line, symbol_type
                )
                # Store original SCIP symbol
                if unified_symbol in self.code_graph.name_to_vertex:
                    vertex_id = self.code_graph.name_to_vertex[unified_symbol]
                    self.code_graph.graph.vs[vertex_id]["scip_symbol"] = symbol
                    self.code_graph.graph.vs[vertex_id]["display_name"] = self._get_display_name(unified_symbol)

                self.code_graph.add_containment_edge(unified_symbol)

                # Update scope for classes and functions
                if symbol_type in [NODE_TYPE_CLASS, NODE_TYPE_FUNCTION, NODE_TYPE_METHOD]:
                    self.code_graph.update_current_scope(
                        unified_symbol, start_line, end_line
                    )
            elif enclosing_ranges and len(enclosing_ranges) >= 4:
                # Use SCIP-provided enclosing ranges
                scope_start_line = int(enclosing_ranges[0])
                scope_end_line = int(enclosing_ranges[2])

                self.code_graph.add_symbol_node(
                    unified_symbol, line, scope_start_line, scope_end_line, symbol_type
                )
                # Store original SCIP symbol
                if unified_symbol in self.code_graph.name_to_vertex:
                    vertex_id = self.code_graph.name_to_vertex[unified_symbol]
                    self.code_graph.graph.vs[vertex_id]["scip_symbol"] = symbol
                    self.code_graph.graph.vs[vertex_id]["display_name"] = self._get_display_name(unified_symbol)

                self.code_graph.add_containment_edge(unified_symbol)

                if symbol_type in [NODE_TYPE_CLASS, NODE_TYPE_FUNCTION, NODE_TYPE_METHOD]:
                    self.code_graph.update_current_scope(
                        unified_symbol, scope_start_line, scope_end_line
                    )
            else:
                # No range information available
                self.code_graph.add_symbol_node(unified_symbol, line, symbol_type=symbol_type)
                # Store original SCIP symbol
                if unified_symbol in self.code_graph.name_to_vertex:
                    vertex_id = self.code_graph.name_to_vertex[unified_symbol]
                    self.code_graph.graph.vs[vertex_id]["scip_symbol"] = symbol
                    self.code_graph.graph.vs[vertex_id]["display_name"] = self._get_display_name(unified_symbol)

                self.code_graph._add_edge(
                    self.code_graph.current_scope, unified_symbol, EDGE_TYPE_CONTAIN
                )

        # Handle reference (not a definition)
        else:
            self.code_graph.add_symbol_reference(unified_symbol, file_path, symbol_type)

    def save_graph(self, output_path):
        """
        Save the code graph to a file.

        Args:
            output_path: Path where the graph should be saved
        """
        self.code_graph.save_graph(output_path)
