import ast
import builtins
import os
from pprint import pprint
from typing import Dict, List

import networkx as nx

from .log_utils import get_logger
from .treetransformer import SymbolTable, SymbolTableBuilder

logger = get_logger(__name__)

# Assuming this is defined elsewhere in your code
exclude_patterns = [
    ".git/",
    "doc/",
    # other patterns as in your original code
]


# Get all built-ins
builtins_list = dir(builtins)

# Filter out specific categories
builtins_functions = [
    name for name in builtins_list if callable(getattr(builtins, name))
]
builtins_constants = [
    name for name in builtins_list if not callable(getattr(builtins, name))
]
builtins_exceptions = [
    name
    for name in builtins_list
    if isinstance(getattr(builtins, name), type)
    and issubclass(getattr(builtins, name), BaseException)
]
all_builtins = builtins_functions + builtins_constants + builtins_exceptions


class ReferenceVisitor(ast.NodeVisitor):
    """Enhanced visitor that can resolve method calls on instances"""

    def __init__(
        self,
        graph: nx.DiGraph,
        function_definitions: Dict[str, List[str]],
        file_path: str,
        symbol_table: SymbolTable,
        symbol_tables: Dict[str, SymbolTable],  # Added parameter
    ):
        self.graph = graph
        self.function_definitions = function_definitions
        self.current_file = file_path
        self.current_class = None
        self.current_function = None
        self.current_scope = file_path
        self.symbol_table: SymbolTable = symbol_table
        self.symbol_tables = symbol_tables  # Access to all symbol tables

    def visit_FunctionDef(self, node):
        function_name = node.name

        if self.current_class:
            full_function_name = (
                f"{self.current_file}::{self.current_class}::{function_name}"
            )
        else:
            full_function_name = f"{self.current_file}::{function_name}"

        prev_function = self.current_function
        self.current_function = full_function_name
        self.current_scope = full_function_name

        self.generic_visit(node)

        # Restore previous context
        self.current_function = prev_function
        self.current_scope = (
            self.current_file
            if not prev_function
            else (
                f"{self.current_file}::{prev_function}"
                if not self.current_class
                else f"{self.current_file}::{self.current_class}::{prev_function}"
            )
        )

    def visit_ClassDef(self, node):
        class_name = node.name
        previous_class = self.current_class  # Save the previous class
        self.current_class = class_name

        self.generic_visit(node)

        # Restore the previous class
        self.current_class = previous_class

    def visit_Call(self, node):
        """Enhanced method to capture both direct function calls and method calls on objects"""

        # Case 1: Direct function call - f()
        if isinstance(node.func, ast.Name):
            self._handle_direct_call(node.func.id)

        # Case 2: Method call on object - obj.method()
        elif isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            self._handle_attribute_call(node.func.value.id, node.func.attr)

        # Handle other cases if needed

        self.generic_visit(node)

    def _handle_direct_call(self, function_name):
        """Handle direct function calls: f()"""
        # Skip built-in functions
        if function_name in all_builtins:
            return

        caller = self.current_function
        if not caller:
            logger.debug(f"No current function when handling call to {function_name}")
            return

        logger.debug(f"Analyzing call to {function_name} in scope {self.current_scope}")

        # Case 1: Local function call in the same file
        if function_name in self.symbol_table.functions:
            # Direct reference to a function in the same file
            callee = self.symbol_table.functions[function_name]
            logger.debug(f"Found local function {function_name} -> {callee}")
            self.graph.add_edge(caller, callee, edge_type="references")
            return

        # Case 2: Imported function or class
        if function_name in self.symbol_table.imports:
            import_info = self.symbol_table.imports[function_name]

            # Skip if this is an external import
            if not import_info.get("is_local", True):
                logger.debug(f"Skipping external import reference: {function_name}")
                return

            imported_path = import_info["path"]
            module_parts = imported_path.split(".")

            # Check if this is a class or function from another module
            if len(module_parts) > 1:
                # module_name = module_parts[0]  # Base module
                imported_name = module_parts[-1]  # Actual entity name

                # Look up the function/class in all symbol tables from other files
                for file_path, sym_table in self.symbol_tables.items():
                    # Check for function
                    if imported_name in sym_table.functions:
                        callee = sym_table.functions[imported_name]
                        logger.debug(
                            f"Found imported function {function_name} -> {callee}"
                        )
                        self.graph.add_edge(caller, callee, edge_type="references")
                        return

                    # Check for class constructor
                    if imported_name in sym_table.classes:
                        # This is a class constructor call
                        class_node = f"{file_path}::{imported_name}"
                        init_method = sym_table.get_method(imported_name, "__init__")
                        if init_method:
                            # Link to constructor if it exists
                            callee = init_method
                        else:
                            # Link to class itself if no constructor
                            callee = class_node
                        logger.debug(
                            f"Found imported class {function_name} -> {callee}"
                        )
                        self.graph.add_edge(caller, callee, edge_type="references")
                        return

        # Case 3: Class reference inside the file
        for class_name, _ in self.symbol_table.classes.items():
            if function_name == class_name:
                # This is a class constructor call, e.g., `x = MyClass()`
                class_node = f"{self.current_file}::{class_name}"
                init_method = self.symbol_table.get_method(class_name, "__init__")
                if init_method:
                    # Link to constructor if it exists
                    callee = init_method
                else:
                    # Link to class itself if no constructor
                    callee = class_node
                logger.debug(f"Found class constructor {function_name} -> {callee}")
                self.graph.add_edge(caller, callee, edge_type="references")
                return

        logger.debug(f"Could not resolve function call: {function_name}")

    def _handle_attribute_call(self, base_name, method_name):
        """Handle method calls on objects: obj.method()"""
        caller = self.current_function
        if not caller:
            return

        # Use the symbol table to resolve the method call
        resolved_method = self.symbol_table.resolve_attribute_call(
            base_name, method_name, self.current_scope
        )
        # debug
        logger.debug(
            f"Resolved method: {resolved_method}, base_name: {base_name}, method_name: {method_name}, scope: {self.current_scope}, file: {self.current_file}"
        )

        if resolved_method:
            self.graph.add_edge(caller, resolved_method, edge_type="method_call")
            return

        # Fallback: Try to find any method with this name in any class
        # This is less accurate but can work when type information is missing
        for class_info in self.symbol_table.classes.values():
            if method_name in class_info["methods"]:
                method_path = class_info["methods"][method_name]
                self.graph.add_edge(
                    caller, method_path, edge_type="possible_method_call"
                )
                # You might want to add a lower confidence score here
                break


class ReferenceBuilder:
    def __init__(self, graph, function_definitions, swe_env=None):
        self.graph = graph
        self.function_definitions = function_definitions
        self.swe_env = swe_env
        self.symbol_tables = {}  # file_path -> SymbolTable

    def build_references(self, repo_path):
        """Build reference edges between functions in the graph."""
        # First pass: Build symbol tables for all files
        for root, _, files in os.walk(repo_path):
            dir_node_name = os.path.relpath(root, repo_path)
            if any(dir_node_name.startswith(pattern) for pattern in exclude_patterns):
                continue

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    rel_file_path = os.path.relpath(file_path, repo_path)
                    if any(
                        rel_file_path.startswith(pattern)
                        for pattern in exclude_patterns
                    ):
                        continue

                    # Build symbol table for this file
                    with open(file_path, "r") as f:
                        try:
                            tree = ast.parse(f.read())
                            symbol_builder = SymbolTableBuilder(
                                rel_file_path, repo_path
                            )
                            symbol_builder.visit(tree)
                            self.symbol_tables[rel_file_path] = (
                                symbol_builder.symbol_table
                            )
                        except SyntaxError:
                            print(f"Syntax error in {file_path}, skipping...")
                            continue

        # Second pass: Analyze function calls with symbol table information
        for root, _, files in os.walk(repo_path):
            dir_node_name = os.path.relpath(root, repo_path)
            if any(dir_node_name.startswith(pattern) for pattern in exclude_patterns):
                continue

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    rel_file_path = os.path.relpath(file_path, repo_path)
                    if any(
                        rel_file_path.startswith(pattern)
                        for pattern in exclude_patterns
                    ):
                        continue

                    # Now analyze with enhanced visitor
                    with open(file_path, "r") as f:
                        try:
                            tree = ast.parse(f.read())
                            symbol_table = self.symbol_tables.get(
                                rel_file_path, SymbolTable()
                            )
                            visitor = ReferenceVisitor(
                                self.graph,
                                self.function_definitions,
                                rel_file_path,
                                symbol_table,
                                self.symbol_tables,  # Pass all symbol tables
                            )
                            visitor.visit(tree)
                        except SyntaxError:
                            continue


def build_graph(repo_path: str) -> nx.DiGraph:
    """
    build_graph analyzes a Python project and builds a graph of class and method relationships.
    """

    print(f"Analyzing project at: {repo_path}")

    # Create a graph to store relationships
    graph = nx.DiGraph()

    # Dictionary to store function definitions
    function_definitions = {}

    # First, collect all function/method definitions in the project
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                rel_file_path = os.path.relpath(
                    file_path, repo_path
                )  # Keep repo_path for relative paths

                # Add file node to graph
                if rel_file_path not in graph:
                    graph.add_node(rel_file_path, type="file")

                try:
                    with open(file_path, "r") as f:
                        tree = ast.parse(f.read())

                        # Build symbol table
                        symbol_builder = SymbolTableBuilder(rel_file_path)
                        symbol_builder.visit(tree)

                        # Register functions and methods in the function_definitions dict
                        for (
                            func_name,
                            func_path,
                        ) in symbol_builder.symbol_table.functions.items():
                            if func_name not in function_definitions:
                                function_definitions[func_name] = []
                            function_definitions[func_name].append(func_path)

                        # Add class nodes and methods to graph
                        for (
                            class_name,
                            class_info,
                        ) in symbol_builder.symbol_table.classes.items():
                            class_node = f"{rel_file_path}::{class_name}"
                            graph.add_node(class_node, type="class")
                            graph.add_edge(
                                rel_file_path, class_node, edge_type="contains"
                            )

                            for _, method_path in class_info["methods"].items():
                                graph.add_node(method_path, type="method")
                                graph.add_edge(
                                    class_node, method_path, edge_type="contains"
                                )

                            # Register methods in function_definitions
                            for method_name in class_info["methods"]:
                                if method_name not in function_definitions:
                                    function_definitions[method_name] = []
                                function_definitions[method_name].append(
                                    class_info["methods"][method_name]
                                )
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Now build the reference edges between functions using ReferenceBuilder
    # Use repo_path instead of repo_path to only analyze files within the project
    reference_builder = ReferenceBuilder(graph, function_definitions)
    reference_builder.build_references(repo_path)  # Changed from repo_path to repo_path

    # Print some statistics
    print(f"Total nodes in graph: {len(graph.nodes)}")
    print(f"Total edges in graph: {len(graph.edges)}")
    print("\nFunction definitions found:")
    pprint(function_definitions)

    # # Save the graph to a file
    # nx.write_json_graph(graph, "graph.json")
    # print("Graph saved to graph.json")

    # Print detected method calls
    print("\nMethod calls detected:")
    method_calls = [
        (u, v)
        for u, v, d in graph.edges(data=True)
        if d.get("edge_type") == "method_call"
    ]
    for caller, callee in method_calls:
        print(f"  {caller} -> {callee}")

    # Print detected references
    print("\nFunction references detected:")
    references = [
        (u, v)
        for u, v, d in graph.edges(data=True)
        if d.get("edge_type") == "references"
    ]
    for caller, callee in references:
        print(f"  {caller} -> {callee}")

    return graph
