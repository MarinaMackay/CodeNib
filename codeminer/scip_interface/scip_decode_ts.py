import re
from pathlib import Path

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


class SCIPTypeScriptGraphDecoder:
    def __init__(self, index_file_path, project_root=None):
        self.index_file_path = index_file_path
        self.project_root = project_root
        self.code_graph = CodeGraph(project_root)
        self.indexed_directories = set()
        self.logger = get_logger(__name__)
        # Register this module for SCIP debug logging
        register_scip_logger(__name__)

        # Track actual file paths from documents to fix index file handling
        self.document_file_paths = set()

    def decode(self):
        self.logger.info(f"Starting SCIP TypeScript decode from {self.index_file_path}")
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
        # Extract file path
        file_match = re.search(r'relative_path:\s*"([^"]+)"', document_text)
        if not file_match:
            return

        file_path = file_match.group(1)
        # Track this file path for index file resolution
        self.document_file_paths.add(file_path)

        # Iteratively Extract the directory path from the file path
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

    def _process_occurrence(self, occurrence_text, file_path):
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

        # Skip stdlib/builtin symbols
        # Check the symbol field specifically, not "scip-typescript" tool name
        if any(
            lib in symbol
            for lib in [" typescript ", "node_modules/", "lib.es", "lib.dom"]
        ):
            return

        # Skip local symbols (scip represents them as 'local <id>')
        if symbol.startswith("local "):
            return

        # Extract symbol_roles
        # IMPORTANT: scip-typescript (v0.4.0) only sets symbol_roles: 1 for definitions
        # References are represented as occurrences WITHOUT the symbol_roles field (value = 0)
        symbol_roles_match = re.search(r"symbol_roles:\s*(\d+)", occurrence_text)
        if symbol_roles_match:
            symbol_roles = int(symbol_roles_match.group(1))
        else:
            # No symbol_roles field means it's a reference (not a definition)
            symbol_roles = 0  # Treat as reference

        # Extract enclosing range if available
        enclosing_ranges = re.findall(r"enclosing_range:\s*(\d+)", occurrence_text)

        # Process the symbol (pass file_path for index handling)
        self._process_symbol(symbol, line, symbol_roles, enclosing_ranges, file_path)

    def _extract_hash(self, symbol_part):
        """
        Extract hash from TypeScript SCIP symbol part.

        Args:
            symbol_part: Symbol part like "add(9b79fb6aee4c0440)" or "Class#method(hash)"

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
            symbol_part: Symbol with hash like "add(hash)" or "Class#method(hash)"

        Returns:
            Symbol without hash: "add" or "Class#method"
        """
        return re.sub(r"\([0-9a-f]+\)", "", symbol_part)

    def _get_display_name(self, unified_symbol):
        """
        Generate human-readable display name from unified symbol.

        Args:
            unified_symbol: Unified symbol like "src/calculator.ts:Calculator.add(hash)"

        Returns:
            Display name like "calculator::Calculator.add"
        """
        # Remove hash for display
        symbol_no_hash = self._remove_hash(unified_symbol)

        # Remove file extension and convert path to module style
        if ":" in symbol_no_hash:
            file_part, symbol_part = symbol_no_hash.split(":", 1)
            # Remove extension
            file_no_ext = re.sub(r"\.(ts|tsx|js|jsx)$", "", file_part)
            # Convert slashes to ::
            module_path = file_no_ext.replace("/", "::")
            return f"{module_path}.{symbol_part}" if symbol_part else module_path
        else:
            return symbol_no_hash

    def _unify_symbol_name(self, symbol, original_symbol, file_path):
        """
        Unify symbol names to a consistent format for TypeScript/JavaScript,
        preserving hash AND package/workspace context for uniqueness.

        Examples:
        - scip-typescript npm @myorg/pkg-a 1.0.0 src/utils/index`/helper().
          -> @myorg/pkg-a@1.0.0:src/utils/index.ts:helper
        - scip-typescript npm pkg-b 1.0.0 src/utils/index`/helper().
          -> pkg-b@1.0.0:src/utils/index.ts:helper
        (Note: Different packages, same path, won't collide)

        Args:
            symbol: Cleaned symbol name (after removing prefix)
            original_symbol: Original full SCIP symbol (for hash and package extraction)
            file_path: Current file path from document (for extension inference)

        Returns:
            Unified symbol name WITH package scope and hash for uniqueness
        """
        # Extract hash from original symbol before any processing
        hash_part = self._extract_hash(original_symbol)

        # Extract package/workspace identifier from original SCIP symbol
        # Format: "scip-typescript <manager> <package> <version> <symbol>"
        # Examples:
        #   "scip-typescript npm @myorg/pkg-a 1.0.0 src/..."
        #   "scip-typescript npm my-package 1.0.0 src/..."
        #   "scip-typescript npm . . src/..." (workspace root)
        package_prefix = None
        parts = original_symbol.split(" ")
        if len(parts) >= 5 and parts[0] == "scip-typescript":
            manager = parts[1]  # npm, yarn, pnpm
            package_name = parts[2]
            version = parts[3]

            # Skip workspace root (. .)
            if package_name != "." and version != ".":
                # Create unique package prefix
                package_prefix = f"{package_name}@{version}"

        # Remove backticks
        clean_symbol = symbol.replace("`", "")

        # Remove hash temporarily for path/name processing
        clean_symbol = self._remove_hash(clean_symbol)

        # Infer file extension from actual file_path instead of guessing
        file_ext = Path(file_path).suffix if file_path else ".ts"
        if not file_ext or file_ext not in [".ts", ".tsx", ".js", ".jsx"]:
            file_ext = ".ts"  # Fallback

        # Parse the symbol path
        # Format: "src/calculator/index`/Calculator#add()."
        if "/" in clean_symbol:
            parts = clean_symbol.split("/", 1)
            module_path = parts[0].replace(".", "/")
            module_path += file_ext

            if len(parts) > 1:
                symbol_part = parts[1]

                # Handle class and method patterns
                if "#" in symbol_part:
                    # Split on # to separate class from method
                    class_method_parts = symbol_part.split("#", 1)
                    class_name = class_method_parts[0]

                    if len(class_method_parts) > 1 and class_method_parts[1]:
                        # Has method after #
                        method_part = class_method_parts[1].rstrip(".")
                        symbol_id = f"{module_path}:{class_name}.{method_part}"
                    else:
                        # Just class (ends with #)
                        symbol_id = f"{module_path}:{class_name}"
                else:
                    # Function (no # symbol)
                    func_name = symbol_part.rstrip(".")
                    symbol_id = f"{module_path}:{func_name}"
            else:
                symbol_id = module_path
        else:
            # No module path separator, use as-is but clean
            symbol_id = clean_symbol.rstrip(".")

        # Build unified symbol with package prefix for cross-package uniqueness
        if package_prefix:
            unified_no_hash = f"{package_prefix}:{symbol_id}"
        else:
            unified_no_hash = symbol_id

        # Add hash back for uniqueness (if present)
        if hash_part:
            unified = f"{unified_no_hash}({hash_part})"
        else:
            unified = unified_no_hash

        return unified

    def _classify_symbol_type(self, unified_symbol, original_symbol=None):
        """
        Classify symbol type based on unified symbol format.

        Args:
            unified_symbol: Unified symbol name
            original_symbol: Original symbol name (for additional context)

        Returns:
            Symbol type: NODE_TYPE_CLASS, NODE_TYPE_METHOD, NODE_TYPE_FIELD, or NODE_TYPE_FUNCTION
        """

        if ":" in unified_symbol:
            symbol_part = unified_symbol.split(":", 1)[1]
            if "." in symbol_part:
                # Has a dot - could be method or field
                # Check if the original symbol had parentheses (indicating method)
                has_parentheses = False
                if original_symbol:
                    has_parentheses = "()" in original_symbol or "(" in original_symbol

                # If original had parentheses, it's a method; otherwise it's a field
                if has_parentheses:
                    return NODE_TYPE_METHOD
                else:
                    return NODE_TYPE_FIELD
            else:
                # Could be class or function - check if it looks like a class
                # In TypeScript, classes typically start with capital letter
                if symbol_part and symbol_part[0].isupper():
                    return NODE_TYPE_CLASS
                else:
                    return NODE_TYPE_FUNCTION
        else:
            return NODE_TYPE_FUNCTION

    def _process_symbol(self, symbol, line, symbol_roles, enclosing_ranges, file_path):
        self.logger.scip_debug(
            f"Processing TS symbol: {symbol} at line {line}, roles: {symbol_roles}, file: {file_path}"
        )

        # Skip function arguments (symbols ending with .(xxx))
        if re.search(r"\.\([^)]+\)$", symbol):
            return

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
            self.logger.error(f"Error exiting scopes at line {line}: {e}")
            raise

        # Clean up the symbol by simply splitting on spaces and taking the last part
        # For example: "scip-typescript npm test-project 1.0.0 src/`calculator.ts`/Calculator#add()."
        # Will become: "src/`calculator.ts`/Calculator#add()."
        # IMPORTANT: Keep original symbol for hash and package extraction
        cleaned_symbol = symbol.split(" ")[-1]
        cleaned_symbol = re.sub(r"`", "", cleaned_symbol)

        # Parse the cleaned symbol
        match = re.search(r"`?([^`]+)`?/([^.]+)(?:\.|\(|#)", cleaned_symbol)
        if not match:
            return

        module_path = match.group(1)

        # Unify symbol name format (pass original symbol for hash/package extraction, file_path for extension)
        unified_symbol = self._unify_symbol_name(cleaned_symbol, symbol, file_path)

        # Classify symbol type (pass both original and unified for context)
        symbol_type = self._classify_symbol_type(unified_symbol, cleaned_symbol)

        # Handle index file exports - use actual file_path instead of constructed path
        # If current file is an index file and symbol has no class/method (#),
        # it's likely a re-export, point reference to this file
        is_index_file = file_path and "/index." in file_path
        symbol_no_hash = self._remove_hash(unified_symbol)

        # Check if symbol is a simple export (no class/method marker)
        is_simple_symbol = "#" not in symbol_no_hash

        if is_index_file and is_simple_symbol:
            # Use bitwise check for reference (not a definition)
            if not (symbol_roles & 1):
                # This is a reference in an index file, likely a re-export
                # Point to the index file itself
                self.code_graph._add_edge(
                    self.code_graph.current_scope, file_path, EDGE_TYPE_REFERENCE
                )
                return

        # Check if this is a definition using bitwise AND (not exact match)
        is_definition = symbol_roles & 1  # Check if definition bit is set

        # Update current scope if this is a definition with enclosing range
        if is_definition and enclosing_ranges and len(enclosing_ranges) >= 4:
            scope_start_line = int(enclosing_ranges[0])
            scope_end_line = int(enclosing_ranges[2])

            # Add symbol node with scope range
            self.code_graph.add_symbol_node(
                unified_symbol, line, scope_start_line, scope_end_line, symbol_type
            )

            # Store original SCIP symbol and display name as node attributes
            if unified_symbol in self.code_graph.name_to_vertex:
                vertex_id = self.code_graph.name_to_vertex[unified_symbol]
                self.code_graph.graph.vs[vertex_id]["scip_symbol"] = symbol
                self.code_graph.graph.vs[vertex_id]["display_name"] = self._get_display_name(unified_symbol)

            # Add containment edge
            self.logger.scip_debug(
                f"Adding containment edge for {unified_symbol}, current scope: {self.code_graph.current_scope}"
            )
            self.code_graph.add_containment_edge(unified_symbol)

            # Update current scope for classes and functions with enclosing ranges
            # Now with proper scope exit handling, functions can safely become scopes
            if symbol_type in [NODE_TYPE_CLASS, NODE_TYPE_FUNCTION, NODE_TYPE_METHOD]:
                try:
                    self.logger.scip_debug(
                        f"Updating scope to {unified_symbol} [{scope_start_line}-{scope_end_line}]"
                    )
                    self.code_graph.update_current_scope(
                        unified_symbol, scope_start_line, scope_end_line
                    )
                except Exception as e:
                    self.logger.error(f"Error updating scope for {unified_symbol}: {e}")
                    raise

        # Handle definition with no enclosing range
        elif is_definition:
            self.logger.scip_debug(
                f"Adding symbol without enclosing range: {unified_symbol}, current scope: {self.code_graph.current_scope}"
            )
            self.code_graph.add_symbol_node(
                unified_symbol, line, symbol_type=symbol_type
            )

            # Store original SCIP symbol and display name
            if unified_symbol in self.code_graph.name_to_vertex:
                vertex_id = self.code_graph.name_to_vertex[unified_symbol]
                self.code_graph.graph.vs[vertex_id]["scip_symbol"] = symbol
                self.code_graph.graph.vs[vertex_id]["display_name"] = self._get_display_name(unified_symbol)

            # Add 'contain' edge from current scope to symbol
            self.code_graph._add_edge(
                self.code_graph.current_scope, unified_symbol, EDGE_TYPE_CONTAIN
            )

        # Handle reference (not a definition)
        # Use bitwise check instead of exact match to handle all reference types
        else:
            self.code_graph.add_symbol_reference(
                unified_symbol, module_path, symbol_type
            )

    def save_graph(self, output_path):
        self.code_graph.save_graph(output_path)
