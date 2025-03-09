import ast
import builtins
import difflib
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
    ):
        self.graph = graph
        self.function_definitions = function_definitions
        self.current_file = file_path
        self.current_class = None
        self.current_function = None
        self.current_scope = file_path
        self.symbol_table: SymbolTable = symbol_table
        self.imports = {}  # Still track imports at this level for compatibility

    def visit_Import(self, node):
        """Track imported modules and their aliases."""
        for alias in node.names:
            imported_name = alias.name  # Full module name
            as_name = alias.asname or imported_name
            self.imports[as_name] = imported_name

        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Track specific imports from a module."""
        module = node.module or ""
        for alias in node.names:
            imported_name = f"{module}.{alias.name}"
            as_name = alias.asname or alias.name
            self.imports[as_name] = imported_name

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        function_name = node.name

        if self.current_class:
            full_function_name = (
                f"{self.current_file}::{self.current_class}::{function_name}"
            )
            self.current_scope = full_function_name
        else:
            full_function_name = f"{self.current_file}::{function_name}"
            self.current_scope = full_function_name

        prev_function = self.current_function
        self.current_function = full_function_name

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

        # Check if it's an imported function/class
        if function_name in self.imports:
            import_module = self.imports[function_name]
            # Handle imported function similar to your original code
            parts = import_module.split(".")
            callable_name = parts[-1]
            module_name = "/".join(parts[:-1])
            compare_str = f"{module_name}::{callable_name}"

            if function_name in self.function_definitions:
                callee_list = self.function_definitions[function_name]
                # Find the most similar function in the callee_list
                callee = difflib.get_close_matches(
                    compare_str, callee_list, n=1, cutoff=0.0
                )
                if callee:
                    callee = callee[0]
                    caller = self.current_function
                    if caller and callee:
                        self.graph.add_edge(caller, callee, edge_type="references")

        # Check local function definitions
        elif function_name in self.function_definitions:
            caller = self.current_function
            callee_list = self.function_definitions[function_name]

            # Prefer functions in the same file
            callee = difflib.get_close_matches(
                self.current_file, callee_list, n=1, cutoff=0.0
            )
            if callee:
                callee = callee[0]
                if caller and callee:
                    self.graph.add_edge(caller, callee, edge_type="references")

    def _handle_attribute_call(self, base_name, method_name):
        """Handle method calls on objects: obj.method()"""
        caller = self.current_function
        if not caller:
            return

        # Use the symbol table to resolve the method call
        resolved_method = self.symbol_table.resolve_attribute_call(
            base_name, method_name, self.current_scope, self.current_file
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
                            symbol_builder = SymbolTableBuilder(rel_file_path)
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

    # Print the symbol tables
    print("\nSymbol tables by file:")
    for file_path, symbol_table in reference_builder.symbol_tables.items():

        if file_path.startswith("my_project"):
            print(f"\nFile: {file_path}")
            print("  Classes:")
            for class_name, class_info in symbol_table.classes.items():
                print(f"    - {class_name}")
                print(f"      Methods: {', '.join(class_info['methods'].keys())}")
                print(f"      Attributes: {', '.join(class_info['attributes'])}")
            print("  Functions:")
            for func_name, _ in symbol_table.functions.items():
                print(f"    - {func_name}")
            print("  Variables with types:")
            for (scope, var_name), var_type in symbol_table.variables.items():
                if scope.startswith("my_project"):
                    print(f"    - {var_name}: {var_type} (in {scope})")

    # # Save the graph to a file
    # nx.write_json_graph(graph, "graph.json")
    # print("Graph saved to graph.json")

    # Print detected method calls
    print("\nMethod calls detected:")
    method_calls = [
        (u, v)
        for u, v, d in graph.edges(data=True)
        if d.get("edge_type") in ("method_call", "possible_method_call")
    ]
    for caller, callee in method_calls:
        print(f"  {caller} -> {callee}")

    return graph
