#!/usr/bin/env python3
"""
SCIP Graph Decoder for C++ projects.

Decodes SCIP index files into CodeGraph representation,
supporting class, method, and function definitions.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

from ..code_chunking import CppCodeChunker
from ..graph.code_graph import CodeGraph
from ..log_utils import get_logger, register_scip_logger
from ..types import (
    EDGE_TYPE_CONTAIN,
    EDGE_TYPE_REFERENCE,
    NODE_TYPE_CLASS,
    NODE_TYPE_FIELD,
    NODE_TYPE_FUNCTION,
    NODE_TYPE_METHOD,
    ROOT_NODE,
)


class SCIPCppGraphDecoder:
    """
    Decoder for C++ SCIP indices.

    Converts binary SCIP index (decoded to text format) into a CodeGraph
    that represents the code structure and relationships.

    Supports:
    - Classes (class/struct definitions)
    - Functions (free functions, including namespace functions)
    - Methods (member functions)
    """

    def __init__(self, index_file_path: str, project_root: Optional[str] = None):
        """
        Initialize the C++ SCIP decoder.

        Args:
            index_file_path: Path to the decoded SCIP index file (text format)
            project_root: Root directory of the project (optional, for file resolution)
        """
        self.index_file_path = index_file_path
        self.project_root = Path(project_root) if project_root else None
        self.code_graph = CodeGraph(project_root)
        self.indexed_directories = set()
        self.logger = get_logger(__name__)
        # Register this module for SCIP debug logging
        register_scip_logger(__name__)

        # Initialize code chunker for finding definitions
        self.code_chunker = CppCodeChunker() if project_root else None

        # Cache for file contents to avoid repeated reads
        self.file_contents_cache = {}

        # Cache for file chunks to avoid repeated chunking
        self.file_chunks_cache = {}

    def decode(self) -> CodeGraph:
        """
        Decode the SCIP index and build the code graph.

        Returns:
            CodeGraph: Populated code graph
        """
        self.logger.info(f"Starting SCIP C++ decode from {self.index_file_path}")
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

    def _process_document(self, document_text: str):
        """
        Process a single document block from the SCIP index.

        Args:
            document_text: Text content of a document block
        """
        # Extract file path
        file_match = re.search(r'relative_path:\s*"([^"]+)"', document_text)
        if not file_match:
            return

        file_path = file_match.group(1)

        # Iteratively extract the directory path from the file path
        dir_path = Path(file_path).parent
        while dir_path != dir_path.parent:  # Stop at the root directory
            dir_path_str = str(dir_path)
            if dir_path_str not in self.indexed_directories:
                # Add directory node if not already indexed
                self.code_graph.add_directory_node(dir_path_str)
                self.indexed_directories.add(dir_path_str)
                # Add containment edge from parent directory to this directory
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

    def _process_occurrence(self, occurrence_text: str, file_path: str):
        """
        Process a single occurrence from the SCIP index.

        Args:
            occurrence_text: Text content of an occurrence block
            file_path: Current file being processed
        """
        # Extract ranges first
        ranges = re.findall(r"range:\s*(\d+)", occurrence_text)
        if len(ranges) < 3:
            return

        line = int(ranges[0])

        # Extract symbol
        symbol_match = re.search(r'symbol:\s*"([^"]+)"', occurrence_text)
        if not symbol_match:
            return

        symbol = symbol_match.group(1)

        # Skip symbols we don't care about
        if not self._should_process_symbol(symbol):
            return

        # Extract symbol_roles
        # symbol_roles: 1 = definition, 0 or missing = reference
        symbol_roles_match = re.search(r"symbol_roles:\s*(\d+)", occurrence_text)
        if symbol_roles_match:
            symbol_roles = int(symbol_roles_match.group(1))
        else:
            # No symbol_roles field means it's a reference
            symbol_roles = 0

        # Extract enclosing range if available
        enclosing_ranges = re.findall(r"enclosing_range:\s*(\d+)", occurrence_text)

        # Process the symbol
        self._process_symbol(symbol, line, symbol_roles, enclosing_ranges, file_path)

    def _should_process_symbol(self, symbol: str) -> bool:
        """
        Determine if a symbol should be processed.

        Args:
            symbol: SCIP symbol string

        Returns:
            bool: True if symbol should be processed
        """
        # Skip local symbols (local variables)
        if symbol.startswith("local "):
            return False

        # Skip file markers
        if "<file>/" in symbol:
            return False

        # Skip macro definitions
        if symbol.endswith("`!"):
            return False

        # Only process C++ symbols (scip-clang format)
        if not symbol.startswith("cxx "):
            return False

        # Skip standard library symbols
        skip_prefixes = ["cxx . . $ std/", "cxx . . $ __"]
        if any(symbol.startswith(prefix) for prefix in skip_prefixes):
            return False

        return True

    def _extract_hash(self, symbol_part: str) -> Optional[str]:
        """
        Extract hash from SCIP symbol part.

        Args:
            symbol_part: Symbol part like "add(9b79fb6aee4c0440)" or "buffered_file#method(hash)"

        Returns:
            Full hash string (16 chars) or None if no hash found
        """
        hash_match = re.search(r"\(([0-9a-f]{8,16})\)", symbol_part)
        if hash_match:
            full_hash = hash_match.group(1)
            return full_hash  # Return full hash to avoid collisions
        return None

    def _remove_hash(self, symbol_part: str) -> str:
        """
        Remove hash from symbol part for display/matching purposes.

        Args:
            symbol_part: Symbol with hash like "add(hash)" or "Class#method(hash)"

        Returns:
            Symbol without hash: "add" or "Class#method"
        """
        return re.sub(r"\([0-9a-f]+\)", "", symbol_part)

    def _get_display_name(self, unified_symbol: str) -> str:
        """
        Generate human-readable display name from unified symbol.

        Args:
            unified_symbol: Unified symbol like "fmt/v6/file#draw(ba5c3d6c)" or "MathUtils/add(hash)"

        Returns:
            Display name like "fmt::v6::file.draw" or "MathUtils::add"
        """
        # Remove hash for display
        symbol_no_hash = self._remove_hash(unified_symbol)

        # Convert SCIP separators to C++ style
        if "#" in symbol_no_hash:
            # Class member: "fmt/v6/file#draw" -> "fmt::v6::file.draw"
            class_path, member_name = symbol_no_hash.split("#", 1)
            class_path = class_path.replace("/", "::")
            if member_name:
                return f"{class_path}.{member_name}"
            else:
                return class_path
        elif "/" in symbol_no_hash:
            # Namespace: "fmt/v6/add" -> "fmt::v6::add"
            return symbol_no_hash.replace("/", "::")
        else:
            return symbol_no_hash

    def _normalize_cpp_symbol(self, symbol: str, file_path: str) -> Tuple[str, str]:
        """
        Normalize C++ SCIP symbol to unified format, preserving hash for uniqueness.

        C++ SCIP symbols follow the format:
        - Namespace/function: "cxx . . $ MathUtils/add(hash)."
        - Class: "cxx . . $ fmt/v6/buffered_file#"
        - Method: "cxx . . $ fmt/v6/buffered_file#buffered_file(hash)."
        - Member: "cxx . . $ fmt/v6/file#width."

        Args:
            symbol: Original SCIP symbol
            file_path: Current file path (stored as node attribute, not in symbol ID)

        Returns:
            Tuple of (unified_symbol, original_part) where:
            - unified_symbol: Normalized symbol WITH hash, like "fmt/v6/file#draw(ba5c3d6c)"
            - original_part: The symbol part after "cxx . . $ "
        """
        # Remove "cxx . . $ " prefix
        if symbol.startswith("cxx . . $ "):
            symbol_part = symbol[10:]  # Remove "cxx . . $ "
        else:
            symbol_part = symbol

        # Remove trailing dots
        symbol_part = symbol_part.rstrip(".")

        # Extract hash (if present)
        hash_part = self._extract_hash(symbol_part)

        # Remove hash temporarily for processing
        symbol_no_hash = self._remove_hash(symbol_part)

        # Build unified symbol keeping SCIP format with hash
        if hash_part:
            # Has hash - append shortened hash
            unified = f"{symbol_no_hash}({hash_part})"
        else:
            # No hash (like classes or fields)
            unified = symbol_no_hash

        return unified, symbol_part

    def _classify_cpp_symbol_type(
        self, unified_symbol: str, original_part: str
    ) -> str:
        """
        Classify C++ symbol type.

        Args:
            unified_symbol: Normalized symbol (e.g., "file.h:Class.method")
            original_part: Original symbol part after "cxx . . $ "

        Returns:
            str: NODE_TYPE_CLASS, NODE_TYPE_METHOD, or NODE_TYPE_FUNCTION
        """
        # Check if it's a class (ends with # in original)
        if "#" in original_part:
            parts = original_part.split("#", 1)
            if len(parts) > 1 and parts[1]:
                # Distinguish field vs method using original SCIP symbol:
                # method has "(" in the part after # (e.g. "Class#method(hash).")
                # field has no "(" (e.g. "Class#field.")
                if "(" in parts[1]:
                    return NODE_TYPE_METHOD
                else:
                    return NODE_TYPE_FIELD
            else:
                # Just class name with #
                return NODE_TYPE_CLASS

        # Otherwise, it's a function
        return NODE_TYPE_FUNCTION

    def _process_symbol(
        self,
        symbol: str,
        line: int,
        symbol_roles: int,
        enclosing_ranges: List[str],
        file_path: str,
    ):
        """
        Process a symbol and add it to the code graph.

        Args:
            symbol: SCIP symbol string
            line: Line number (0-based from SCIP)
            symbol_roles: Symbol roles bitset (1=definition, 0=reference)
            enclosing_ranges: Enclosing range for scope (list of line numbers)
            file_path: Current file path
        """
        self.logger.scip_debug(
            f"Processing C++ symbol: {symbol} at line {line}, roles: {symbol_roles}"
        )

        # Normalize the symbol
        unified_symbol, original_part = self._normalize_cpp_symbol(symbol, file_path)

        # Classify symbol type
        symbol_type = self._classify_cpp_symbol_type(unified_symbol, original_part)

        # Exit scopes that have ended based on current line
        try:
            self.logger.scip_debug(
                f"Scope stack before exit: {[list(s.keys())[0] for s in self.code_graph.scope_stack]}"
            )
            self.code_graph.exit_scopes_by_line(line)
            self.logger.scip_debug(
                f"Scope stack after exit: {[list(s.keys())[0] for s in self.code_graph.scope_stack]}"
            )
        except Exception as e:
            self.logger.error(f"Error exiting scopes: {e}")

        # Convert 0-based line to 1-based for graph
        line_1based = line + 1

        # Check if this is a definition or reference
        is_definition = symbol_roles & 1  # Check if definition bit is set

        if is_definition:
            # This is a definition
            # Try to find definition range using code chunker first
            definition_range = None
            if symbol_type in [NODE_TYPE_CLASS, NODE_TYPE_METHOD, NODE_TYPE_FUNCTION]:
                # Use unified_symbol for lookup (it's in the format chunker expects)
                definition_range = self._find_definition_range(file_path, unified_symbol, line_1based)

            if definition_range:
                # Use code chunker result
                scope_start_line, scope_end_line = definition_range
                self.logger.scip_debug(
                    f"Definition with chunker range: {unified_symbol} "
                    f"at line {line_1based}, scope: {scope_start_line}-{scope_end_line}"
                )

                self.code_graph.add_symbol_node(
                    unified_symbol,
                    line_1based,
                    scope_start_line,
                    scope_end_line,
                    symbol_type,
                )
                # Store original SCIP symbol and display name as node attributes
                if unified_symbol in self.code_graph.name_to_vertex:
                    vertex_id = self.code_graph.name_to_vertex[unified_symbol]
                    self.code_graph.graph.vs[vertex_id]["scip_symbol"] = symbol
                    self.code_graph.graph.vs[vertex_id]["display_name"] = self._get_display_name(unified_symbol)

                if symbol_type in [NODE_TYPE_CLASS, NODE_TYPE_FUNCTION, NODE_TYPE_METHOD]:
                    self.code_graph.update_current_scope(
                        unified_symbol, scope_start_line, scope_end_line
                    )
            elif enclosing_ranges and len(enclosing_ranges) >= 3:
                # Has enclosing range (scope information) from SCIP
                scope_start_line = int(enclosing_ranges[0])
                scope_end_line = int(enclosing_ranges[2])

                self.logger.scip_debug(
                    f"Definition with SCIP scope: {unified_symbol} "
                    f"at line {line_1based}, scope: {scope_start_line}-{scope_end_line}"
                )

                # Add the symbol to the graph
                self.code_graph.add_symbol_node(
                    unified_symbol,
                    line_1based,
                    scope_start_line,
                    scope_end_line,
                    symbol_type,
                )
                # Store original SCIP symbol and display name
                if unified_symbol in self.code_graph.name_to_vertex:
                    vertex_id = self.code_graph.name_to_vertex[unified_symbol]
                    self.code_graph.graph.vs[vertex_id]["scip_symbol"] = symbol
                    self.code_graph.graph.vs[vertex_id]["display_name"] = self._get_display_name(unified_symbol)

                # Update current scope
                if symbol_type in [NODE_TYPE_CLASS, NODE_TYPE_FUNCTION, NODE_TYPE_METHOD]:
                    self.code_graph.update_current_scope(
                        unified_symbol, scope_start_line, scope_end_line
                    )
            else:
                # Definition without enclosing range
                # Likely a class or top-level function
                self.logger.scip_debug(
                    f"Definition without scope: {unified_symbol} at line {line_1based}"
                )

                # Add the symbol to the graph without scope
                self.code_graph.add_symbol_node(
                    unified_symbol,
                    line_1based,
                    symbol_type=symbol_type,
                )
                # Store original SCIP symbol and display name
                if unified_symbol in self.code_graph.name_to_vertex:
                    vertex_id = self.code_graph.name_to_vertex[unified_symbol]
                    self.code_graph.graph.vs[vertex_id]["scip_symbol"] = symbol
                    self.code_graph.graph.vs[vertex_id]["display_name"] = self._get_display_name(unified_symbol)

                # Add containment edge if we have a current scope
                if self.code_graph.current_scope:
                    self.code_graph._add_edge(
                        self.code_graph.current_scope, unified_symbol, EDGE_TYPE_CONTAIN
                    )
        else:
            # This is a reference
            self.logger.scip_debug(
                f"Reference: {unified_symbol} at line {line_1based}"
            )

            # Add reference edge
            self.code_graph.add_symbol_reference(
                unified_symbol, file_path, symbol_type
            )

    def _get_file_chunks(self, file_path: str):
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

    def _find_definition_range(self, file_path: str, symbol_name: str, line: int) -> Optional[Tuple[int, int]]:
        """
        Find the definition range for a symbol using the C++ code chunker.

        Args:
            file_path: Path to the file containing the symbol
            symbol_name: Name of the symbol (with hash like "fmt/v6/file#draw(ba5c3d6c)")
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
        # SCIP format: "fmt/v6/file#draw" -> "fmt::v6::file.draw"
        display_name = self._get_display_name(symbol_no_hash)

        # Extract simple name and various possible chunker formats
        if "." in display_name:
            # Method: "fmt::v6::Class.method" -> try "method", "Class::method", full name
            parts = display_name.split(".")
            full_class_path = parts[0]  # e.g., "fmt::v6::Class"
            method_name = parts[-1]  # e.g., "method"

            # Get simple class name (last component)
            class_name = full_class_path.split("::")[-1]  # e.g., "Class"

            possible_names = [
                method_name,  # Simple name
                f"{class_name}::{method_name}",  # Class::method
                f"{full_class_path}::{method_name}",  # Full qualified name with ::
                display_name,  # Original with .
            ]
        elif "::" in display_name:
            # Namespace function: "fmt::v6::add" -> try "add", partial paths, full path
            components = display_name.split("::")
            simple_name = components[-1]

            # Generate possible names with different namespace depths
            possible_names = [simple_name, display_name]
            # Add intermediate qualified names
            for i in range(len(components) - 1, 0, -1):
                possible_names.append("::".join(components[i:]))

        else:
            # Simple name (class or function)
            possible_names = [display_name]

        # Look for the symbol in chunks
        for chunk in chunks:
            chunk_name = getattr(chunk, "name", "")
            if not chunk_name:
                continue

            # Try exact matches first
            if chunk_name in possible_names:
                start_line = getattr(chunk, "start_line", 0)
                end_line = getattr(chunk, "end_line", start_line)
                return (start_line, end_line)

            # Try suffix matches for qualified names
            for name in possible_names:
                if chunk_name.endswith(f"::{name}") or chunk_name == name:
                    start_line = getattr(chunk, "start_line", 0)
                    end_line = getattr(chunk, "end_line", start_line)
                    return (start_line, end_line)

        return None

    def save_graph(self, output_path: str):
        """
        Save the code graph to a file.

        Args:
            output_path: Path where the graph should be saved
        """
        self.code_graph.save_graph(output_path)
