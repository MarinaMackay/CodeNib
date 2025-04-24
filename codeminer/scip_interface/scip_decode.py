import re

from ..code_graph import CodeGraph


class SCIPGraphDecoder:
    def __init__(self, index_file_path):
        self.index_file_path = index_file_path
        self.code_graph = CodeGraph()

    def decode(self):
        with open(self.index_file_path, "r") as f:
            content = f.read()

        # Parse documents
        document_blocks = re.findall(
            r"documents\s*{(.*?)(?=documents\s*{|$)", content, re.DOTALL
        )

        for document in document_blocks:
            self._process_document(document)

        return self.code_graph

    def _process_document(self, document_text):
        # Extract file path
        file_match = re.search(r'relative_path:\s*"([^"]+)"', document_text)
        if not file_match:
            return

        file_path = file_match.group(1)

        # Add file node
        self.code_graph.add_file_node(file_path)

        # Process occurrences
        occurrences = re.findall(r"occurrences\s*{(.*?)}", document_text, re.DOTALL)
        for occurrence in occurrences:
            self._process_occurrence(occurrence)

    def _process_occurrence(self, occurrence_text):
        # Skip local symbols
        if "local" in occurrence_text:
            return

        # Skip stdlib symbols
        if "python-stdlib" in occurrence_text:
            return

        # Extract ranges
        ranges = re.findall(r"range:\s*(\d+)", occurrence_text)
        if len(ranges) < 3:
            return

        line = int(ranges[0])

        # Extract symbol
        symbol_match = re.search(r'symbol:\s*"([^"]+)"', occurrence_text)
        if not symbol_match:
            return

        symbol = symbol_match.group(1)

        # Extract symbol_roles
        symbol_roles_match = re.search(r"symbol_roles:\s*(\d+)", occurrence_text)
        if not symbol_roles_match:
            return

        symbol_roles = int(symbol_roles_match.group(1))

        # Extract enclosing range if available
        enclosing_ranges = re.findall(r"enclosing_range:\s*(\d+)", occurrence_text)

        # Process the symbol
        self._process_symbol(symbol, line, symbol_roles, enclosing_ranges)

    def _process_symbol(self, symbol, line, symbol_roles, enclosing_ranges):
        # Skip function arguments (symbols ending with .(xxx))
        if re.search(r"\.\([^)]+\)$", symbol):
            return

        # Parse the symbol
        match = re.search(r"`?([^`]+)`?/([^.]+)(?:\.|\(|#)", symbol)
        if not match:
            return

        module_path = match.group(1)

        # Clean up the symbol by simply splitting on spaces and taking the last part
        # For example: "scip-python python HttpieCliRepo 5b604c37c6c67e18e7c3e9aee6c88a8c22b98345 extras.profiling.benchmarks/QuietSimpleHTTPServer#log_message()."
        # Will become: "extras.profiling.benchmarks/QuietSimpleHTTPServer#log_message()."
        cleaned_symbol = symbol.split(" ")[-1]
        cleaned_symbol = re.sub(r"`", "", cleaned_symbol)

        # Handle __init__ symbols - convert to file reference
        if "/__init__" in cleaned_symbol:
            # Extract the module path and use it as the target
            module_match = re.search(r"(.+)/(?:__init__)", cleaned_symbol)
            if module_match:
                module_path = module_match.group(1)
                file_path = module_path.replace(".", "/") + ".py"

                # If this is a reference, point to the file instead
                if symbol_roles == 8:
                    self.code_graph._add_edge(
                        self.code_graph.current_scope, file_path, "reference"
                    )
                return

        # Update current scope if this is a definition with enclosing range
        if symbol_roles == 1 and enclosing_ranges and len(enclosing_ranges) >= 4:
            scope_start_line = int(enclosing_ranges[0])
            scope_end_line = int(enclosing_ranges[2])

            # Add symbol node with scope range
            self.code_graph.add_symbol_node(
                cleaned_symbol, line, scope_start_line, scope_end_line
            )

            # Add containment edge
            self.code_graph.add_containment_edge(cleaned_symbol)

            # Update current scope
            self.code_graph.update_current_scope(cleaned_symbol)

        # Handle definition (symbol_roles == 1) with no enclosing range
        elif symbol_roles == 1:
            self.code_graph.add_symbol_node(cleaned_symbol, line)

            # Add 'contain' edge from current scope to symbol
            self.code_graph._add_edge(
                self.code_graph.current_scope, cleaned_symbol, "contain"
            )

        # Handle reference (symbol_roles == 8)
        elif symbol_roles == 8:
            self.code_graph.add_symbol_reference(cleaned_symbol, module_path)

    def save_graph(self, output_path):
        self.code_graph.save_graph(output_path)
