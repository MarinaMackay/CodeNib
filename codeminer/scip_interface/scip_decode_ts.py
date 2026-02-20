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

        # Track project's own SCIP package names to filter external refs
        self._project_packages = set()

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

        self.logger.info(
            f"Decoded {len(document_blocks)} documents, "
            f"nodes={self.code_graph.graph.vcount()}, "
            f"edges={self.code_graph.graph.ecount()}"
        )

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

        # Extract symbol_roles early (needed for external-package filtering)
        symbol_roles_match = re.search(r"symbol_roles:\s*(\d+)", occurrence_text)
        symbol_roles = int(symbol_roles_match.group(1)) if symbol_roles_match else 0

        # Filter external / third-party package references.
        # SCIP symbol: "scip-typescript npm <pkg> <ver> <descriptor>"
        scip_parts = symbol.split(" ")
        if len(scip_parts) >= 5 and scip_parts[0] == "scip-typescript":
            pkg_name = scip_parts[2]
            pkg_version = scip_parts[3]
            if symbol_roles & 1:
                # Record project packages from definitions
                if pkg_name != "." and pkg_version != ".":
                    self._project_packages.add(pkg_name)
            else:
                # Always filter @types/* references
                if pkg_name.startswith("@types/"):
                    return
                # Filter non-project packages once we know our own
                if (
                    self._project_packages
                    and pkg_name != "."
                    and pkg_name not in self._project_packages
                ):
                    return

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

        Format: ``package@ver:file/path.ext:Symbol`` →
                ``package@ver.file/path.ext:Symbol``
        """
        # Remove hash for display
        symbol_no_hash = self._remove_hash(unified_symbol)

        # Clean trailing digit+colon suffix:  isArray0: -> isArray
        symbol_no_hash = re.sub(r"\d+:$", "", symbol_no_hash)
        symbol_no_hash = symbol_no_hash.rstrip(":")

        # Replace the first ':' with '.' for readability
        if ":" in symbol_no_hash:
            first_colon = symbol_no_hash.index(":")
            display = symbol_no_hash[:first_colon] + "." + symbol_no_hash[first_colon + 1:]
        else:
            display = symbol_no_hash

        return display

    def _unify_symbol_name(self, symbol, original_symbol, file_path):
        """
        Unify symbol names to a consistent format for TypeScript/JavaScript,
        preserving hash AND package/workspace context for uniqueness.

        Uses backtick boundaries in the SCIP symbol to extract the correct
        file path (e.g. ``lib/utils.js`` from ```lib/utils.js```).

        Args:
            symbol: Cleaned symbol name (after removing prefix)
            original_symbol: Original full SCIP symbol (for hash and package extraction)
            file_path: Current file path from document (for extension inference)

        Returns:
            Unified symbol name WITH package scope and hash for uniqueness
        """
        hash_part = self._extract_hash(original_symbol)

        # --- Package prefix ---
        package_prefix = None
        scip_parts = original_symbol.split(" ")
        if len(scip_parts) >= 5 and scip_parts[0] == "scip-typescript":
            package_name = scip_parts[2]
            version = scip_parts[3]
            if package_name != "." and version != ".":
                package_prefix = f"{package_name}@{version}"
            # Symbol descriptor with backticks intact
            sym_with_bt = " ".join(scip_parts[4:])
        else:
            sym_with_bt = original_symbol.split(" ")[-1]

        # --- Extract file path from backtick boundaries ---
        # SCIP format: `file/path.js`/SymbolDescriptor
        all_bt_segments = re.findall(r"`([^`]+)`", sym_with_bt)

        if all_bt_segments:
            if len(all_bt_segments) == 1:
                module_path = all_bt_segments[0]
            else:
                # Multi-segment: first segment may use dots for slashes
                first = re.sub(r"\.(ts|tsx|js|jsx)$", "", all_bt_segments[0])
                first = first.replace(".", "/")
                module_path = first + "/" + "/".join(all_bt_segments[1:])

            # Everything after the last closing backtick is the descriptor
            last_bt_end = sym_with_bt.rfind("`") + 1
            symbol_descriptor = sym_with_bt[last_bt_end:].lstrip("/")
        else:
            # Fallback: no backticks — use the document file path
            module_path = file_path or "unknown"
            symbol_descriptor = self._remove_hash(symbol.replace("`", ""))

        # Clean descriptor
        symbol_descriptor = self._remove_hash(symbol_descriptor).rstrip()

        # --- Build symbol_id ---
        if symbol_descriptor:
            if "#" in symbol_descriptor:
                class_method = symbol_descriptor.split("#", 1)
                class_name = class_method[0]
                if class_method[1]:
                    method_part = class_method[1].rstrip(".")
                    symbol_id = f"{module_path}:{class_name}.{method_part}"
                else:
                    symbol_id = f"{module_path}:{class_name}"
            else:
                name = symbol_descriptor.rstrip(".")
                symbol_id = f"{module_path}:{name}"
        else:
            # Module-level export (empty descriptor, just `file`/)
            symbol_id = module_path

        # --- Assemble unified name ---
        if package_prefix:
            unified_no_hash = f"{package_prefix}:{symbol_id}"
        else:
            unified_no_hash = symbol_id

        if hash_part:
            return f"{unified_no_hash}({hash_part})"
        return unified_no_hash

    def _classify_symbol_type(self, unified_symbol, original_symbol=None):
        """
        Classify symbol type based on the SCIP symbol descriptor suffix.

        SCIP encodes the symbol kind in the trailing descriptor characters:
          - ``().``  → method (if after ``#``) or function
          - ``#``    → type definition (class / interface / enum / type alias)
          - ``.``    → field / property / variable

        Args:
            unified_symbol: Unified symbol name
            original_symbol: Original cleaned SCIP symbol (with descriptor suffixes)

        Returns:
            Symbol type constant
        """
        if not original_symbol:
            return NODE_TYPE_FIELD

        # Strip hash before checking suffix so that (hash). becomes ().
        sym = self._remove_hash(original_symbol).rstrip()

        # "Foo#bar()." or "bar()." → method / function
        if sym.endswith("()."):
            # If there's a '#' before the method name it's a class method
            # Strip the trailing method part and check
            base = sym[:-3]  # remove "()."
            # Find the last path separator
            last_sep = max(base.rfind("/"), base.rfind("#"))
            if last_sep != -1 and base[last_sep] == "#":
                return NODE_TYPE_METHOD
            return NODE_TYPE_FUNCTION

        # "Foo#" → class / interface / type
        if sym.endswith("#"):
            return NODE_TYPE_CLASS

        # "Foo#bar." → field / property
        if sym.endswith("."):
            return NODE_TYPE_FIELD

        return NODE_TYPE_FIELD

    def _process_symbol(self, symbol, line, symbol_roles, enclosing_ranges, file_path):
        self.logger.scip_debug(
            f"Processing TS symbol: {symbol} at line {line}, "
            f"roles: {symbol_roles}, file: {file_path}"
        )

        # Skip function arguments (symbols ending with .(xxx))
        if re.search(r"\.\([^)]+\)$", symbol):
            return

        # Exit scopes that have ended based on current line
        try:
            scope_stack_names = [list(s.keys())[0] for s in self.code_graph.scope_stack]
            self.logger.scip_debug(f"Scope stack before exit: {scope_stack_names}")
            self.code_graph.exit_scopes_by_line(line)
            scope_stack_names = [list(s.keys())[0] for s in self.code_graph.scope_stack]
            self.logger.scip_debug(f"Scope stack after exit: {scope_stack_names}")
        except Exception as e:
            self.logger.error(f"Error exiting scopes at line {line}: {e}")
            raise

        # Clean up the symbol by simply splitting on spaces and taking the last part
        cleaned_symbol = symbol.split(" ")[-1]
        cleaned_symbol = re.sub(r"`", "", cleaned_symbol)

        # Parse the cleaned symbol
        match = re.search(r"`?([^`]+)`?/([^.]+)(?:\.|\(|#)", cleaned_symbol)
        if not match:
            return

        # Unify symbol name format
        unified_symbol = self._unify_symbol_name(cleaned_symbol, symbol, file_path)

        # Filter generic type parameters like .[T], .[D]
        if re.search(r"\.\[[A-Z]\w*\]", unified_symbol):
            return

        # Filter module-level export objects (bare file path, no symbol descriptor)
        sym_after_pkg = (
            unified_symbol.split(":", 1)[-1]
            if ":" in unified_symbol
            else unified_symbol
        )
        if re.match(r"^[^:]+\.(js|ts|jsx|tsx)$", sym_after_pkg):
            return

        # Classify symbol type
        symbol_type = self._classify_symbol_type(unified_symbol, cleaned_symbol)

        # Handle index file exports
        is_index_file = file_path and "/index." in file_path
        symbol_no_hash = self._remove_hash(unified_symbol)
        is_simple_symbol = "#" not in symbol_no_hash

        if is_index_file and is_simple_symbol:
            if not (symbol_roles & 1):
                if self.code_graph.current_scope != file_path:
                    self.code_graph._add_edge(
                        self.code_graph.current_scope, file_path, EDGE_TYPE_REFERENCE
                    )
                return

        # Check if this is a definition using bitwise AND
        is_definition = symbol_roles & 1

        # Update current scope if this is a definition with enclosing range
        if is_definition and enclosing_ranges and len(enclosing_ranges) >= 4:
            scope_start_line = int(enclosing_ranges[0])
            scope_end_line = int(enclosing_ranges[2])

            # Add symbol node with scope range
            self.code_graph.add_symbol_node(
                unified_symbol, line, scope_start_line, scope_end_line, symbol_type
            )

            # Store display name as node attribute
            if unified_symbol in self.code_graph.name_to_vertex:
                vertex_id = self.code_graph.name_to_vertex[unified_symbol]

                self.code_graph.graph.vs[vertex_id]["display_name"] = (
                    self._get_display_name(unified_symbol)
                )

            # Add containment edge
            self.logger.scip_debug(
                f"Adding containment edge for {unified_symbol}, "
                f"current scope: {self.code_graph.current_scope}"
            )
            self.code_graph.add_containment_edge(unified_symbol)

            # Update current scope for classes and functions with enclosing ranges
            if symbol_type in [NODE_TYPE_CLASS, NODE_TYPE_FUNCTION, NODE_TYPE_METHOD]:
                try:
                    self.logger.scip_debug(
                        f"Updating scope to {unified_symbol} "
                        f"[{scope_start_line}-{scope_end_line}]"
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
                f"Adding symbol without enclosing range: {unified_symbol}, "
                f"current scope: {self.code_graph.current_scope}"
            )
            self.code_graph.add_symbol_node(
                unified_symbol, line, symbol_type=symbol_type
            )

            # Store display name
            if unified_symbol in self.code_graph.name_to_vertex:
                vertex_id = self.code_graph.name_to_vertex[unified_symbol]

                self.code_graph.graph.vs[vertex_id]["display_name"] = (
                    self._get_display_name(unified_symbol)
                )

            # Add 'contain' edge from current scope to symbol
            self.code_graph._add_edge(
                self.code_graph.current_scope, unified_symbol, EDGE_TYPE_CONTAIN
            )

        # Handle reference (not a definition)
        # Use bitwise check instead of exact match to handle all reference types
        else:
            self.code_graph.add_symbol_reference(unified_symbol, file_path, symbol_type)
            # Set display_name for reference-only nodes (first occurrence wins)
            if unified_symbol in self.code_graph.name_to_vertex:
                vid = self.code_graph.name_to_vertex[unified_symbol]
                if not self.code_graph.graph.vs[vid].attributes().get("display_name"):
                    self.code_graph.graph.vs[vid]["display_name"] = (
                        self._get_display_name(unified_symbol)
                    )

    def save_graph(self, output_path):
        self.code_graph.save_graph(output_path)
