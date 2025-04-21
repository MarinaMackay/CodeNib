import json

import igraph as ig


class CodeGraph:
    """
    A class to represent and manipulate a code graph using igraph.
    """

    def __init__(self):
        # Create a directed graph
        self.graph = ig.Graph(directed=True)
        self.current_file = None
        self.current_scope = None
        self.scope_stack = []
        # Store line ranges for symbols
        self.symbol_ranges = {}
        # Map symbol names to vertex IDs
        self.name_to_vertex = {}

    def add_file_node(self, file_path):
        """
        Add a file node to the graph and set it as the current file and scope.

        Args:
            file_path: Path of the file to add
        """
        self.current_file = file_path

        # Add vertex for file
        self._add_vertex(file_path, {"type": "file"})

        self.current_scope = file_path
        self.scope_stack = [file_path]

    def add_symbol_node(self, symbol, line, scope_start_line=None, scope_end_line=None):
        """
        Add a symbol node to the graph.

        Args:
            symbol: Symbol name
            line: Line number of the symbol
            scope_start_line: Start line of the symbol's scope (optional)
            scope_end_line: End line of the symbol's scope (optional)
        """
        if scope_start_line and scope_end_line:
            # Store symbol range
            self.symbol_ranges[symbol] = (scope_start_line, scope_end_line)

            # Add symbol vertex with scope range
            self._add_vertex(
                symbol,
                {
                    "type": "symbol",
                    "file": self.current_file,
                    "start_line": scope_start_line,
                    "end_line": scope_end_line,
                },
            )
        else:
            # Add symbol vertex without scope range
            self._add_vertex(
                symbol,
                {
                    "type": "symbol",
                    "file": self.current_file,
                    "start_line": line,
                    "end_line": line,
                },
            )

    def add_symbol_reference(self, symbol, module_path=None):
        """
        Add a reference to a symbol.

        Args:
            symbol: Symbol being referenced
            module_path: Path of the module containing the symbol (optional)
        """
        # If the symbol doesn't exist, create it without range info
        if symbol not in self.name_to_vertex:
            file_attr = module_path if module_path else None
            self._add_vertex(symbol, {"type": "symbol", "file": file_attr})

        # Add reference edge
        self._add_edge(self.current_scope, symbol, "reference")

    def update_current_scope(self, symbol):
        """
        Update the current scope to the given symbol.

        Args:
            symbol: Symbol to set as current scope
        """
        self.current_scope = symbol
        self.scope_stack.append(symbol)

    def add_containment_edge(self, target_symbol):
        """
        Add a containment edge from current scope to a symbol.

        Args:
            target_symbol: Symbol being contained
        """
        parent_scope = (
            self.scope_stack[-2] if len(self.scope_stack) > 1 else self.current_file
        )
        self._add_edge(parent_scope, target_symbol, "contain")

    def _add_vertex(self, name, attributes=None):
        """
        Add a vertex to the graph if it doesn't exist.

        Args:
            name: Name of the vertex
            attributes: Dictionary of vertex attributes

        Returns:
            Vertex ID
        """
        if name in self.name_to_vertex:
            vertex_id = self.name_to_vertex[name]
            # Update attributes if provided
            if attributes:
                for key, value in attributes.items():
                    self.graph.vs[vertex_id][key] = value
            return vertex_id

        # Add a new vertex
        self.graph.add_vertices(1)
        vertex_id = self.graph.vcount() - 1
        self.name_to_vertex[name] = vertex_id

        # Set the name attribute
        self.graph.vs[vertex_id]["name"] = name

        # Set other attributes if provided
        if attributes:
            for key, value in attributes.items():
                self.graph.vs[vertex_id][key] = value

        return vertex_id

    def _add_edge(self, source_name, target_name, edge_type):
        """
        Add an edge between two vertices.

        Args:
            source_name: Name of the source vertex
            target_name: Name of the target vertex
            edge_type: Type of the edge (e.g., "reference", "contain")

        Returns:
            Edge ID
        """
        # Make sure both vertices exist
        source_id = (
            self._add_vertex(source_name)
            if source_name not in self.name_to_vertex
            else self.name_to_vertex[source_name]
        )
        target_id = (
            self._add_vertex(target_name)
            if target_name not in self.name_to_vertex
            else self.name_to_vertex[target_name]
        )

        # Add edge
        self.graph.add_edges([(source_id, target_id)])
        edge_id = self.graph.ecount() - 1

        # Set edge type
        self.graph.es[edge_id]["type"] = edge_type

        return edge_id

    def save_graph(self, output_path):
        """
        Save the graph to a JSON file.

        Args:
            output_path: Path to save the JSON file
        """
        # Convert graph to JSON
        data = {"nodes": [], "edges": []}

        # Add nodes
        for vertex in self.graph.vs:
            node_data = {"id": vertex["name"]}
            # Add all other attributes
            for key in vertex.attributes():
                if key != "name":  # "id" is already set from "name"
                    node_data[key] = vertex[key]
            data["nodes"].append(node_data)

        # Add edges
        for edge in self.graph.es:
            source_name = self.graph.vs[edge.source]["name"]
            target_name = self.graph.vs[edge.target]["name"]
            edge_data = {
                "source": source_name,
                "target": target_name,
                "type": edge["type"] if "type" in edge.attributes() else None,
            }
            data["edges"].append(edge_data)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_graph(self):
        """
        Get the igraph Graph object.

        Returns:
            The igraph Graph instance
        """
        return self.graph
