import ast
import builtins
import os

# from pprint import pprint
from typing import Dict

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
        file_path: str,
        symbol_table: SymbolTable,
        symbol_tables: Dict[str, SymbolTable],  # Added parameter
    ):
        self.graph = graph
        self.current_file = file_path
        self.current_class = None
        self.current_function = None
        self.current_scope = file_path
        self.symbol_table: SymbolTable = symbol_table
        self.symbol_tables = symbol_tables  # Access to all symbol tables

    def visit_FunctionDef(self, node):
        function_name = node.name
        prev_function = self.current_function
        prev_scope = self.current_scope

        if self.current_class:
            # This is a method in a class
            self.current_function = (
                f"{self.current_file}::{self.current_class}::{function_name}"
            )
        elif self.current_function:
            # This is a nested function
            # however, we use the parent function as the function
            # log
            logger.info(
                f"Reference: Nested function found: {function_name} in {self.current_function}"
            )
        else:
            # This is a top-level function
            self.current_function = f"{self.current_file}::{function_name}"
        self.current_scope = self.current_function

        self.generic_visit(node)

        # Restore previous context
        self.current_function = prev_function
        self.current_scope = prev_scope

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
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                self._handle_attribute_call(node.func.value.id, node.func.attr)
            elif isinstance(node.func.value, ast.Attribute):
                self._handle_nested_attribute_call(node.func.value, node.func.attr)

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
            # TODO, this is global call from the file
            return

        # logger.debug(f"Analyzing call to {function_name} in scope {self.current_scope}")

        # Case 1: Local function call in the same file
        if function_name in self.symbol_table.functions:
            # Direct reference to a function in the same file
            callee = self.symbol_table.functions[function_name]
            # logger.debug(f"Found local function {function_name} -> {callee}")
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
                        # logger.debug(
                        #     f"Found imported function {function_name} -> {callee}"
                        # )
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
                        # logger.debug(
                        #     f"Found imported class {function_name} -> {callee}"
                        # )
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
                # logger.debug(f"Found class constructor {function_name} -> {callee}")
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
            base_name, method_name, self.current_scope, self.symbol_tables
        )
        # debug
        # logger.debug(
        #     f"Resolved method: {resolved_method}, base_name: {base_name}, method_name: {method_name}, scope: {self.current_scope}, file: {self.current_file}"
        # )

        if resolved_method:
            self.graph.add_edge(caller, resolved_method, edge_type="method_call")
            return

    def _handle_nested_attribute_call(self, attr_value, method_name):
        """Handle nested attribute calls like self.obj.method() or a.b.method()"""
        caller = self.current_function
        if not caller:
            return

        # Build the attribute chain (e.g., self.obj.attr would be ["self", "obj"])
        attr_chain = []
        current = attr_value

        # Walk up the chain of attributes
        while isinstance(current, ast.Attribute):
            attr_chain.insert(0, current.attr)
            current = current.value

        # Add the base object at the start
        if isinstance(current, ast.Name):
            attr_chain.insert(0, current.id)
        else:
            # If we can't identify the base, we can't resolve the call
            return

        logger.info(f"Nested attribute call: {attr_chain} -> {method_name}")

        # Special case for self.x.y.method()
        if attr_chain[0] == "self" and self.current_class:
            current_type = self.current_class
            current_file = self.current_file
            current_scope = f"{current_file}::{self.current_class}"
            current_sym_table = self.symbol_table

            # Initialize a chain to track our path
            type_chain = [current_type]
            file_path_chain = [current_file]

            # Start from the first attribute after "self"
            for i in range(1, len(attr_chain)):
                attr_name = attr_chain[i]
                obj_type = None

                # For the first attribute (immediately after 'self')
                if i == 1:
                    # Try current class scope
                    obj_type = current_sym_table.get_variable_type(
                        attr_name, current_scope
                    )

                    # If not found, try the class scope directly
                    if not obj_type:
                        class_scope = f"{current_file}::{self.current_class}"
                        obj_type = current_sym_table.get_variable_type(
                            attr_name, class_scope
                        )
                else:
                    # For subsequent attributes, look only in the current type's scope
                    if "::" in current_type:
                        # If fully qualified type (file::class)
                        parts = current_type.split("::")
                        type_file_path = parts[0]
                        class_name = parts[1]

                        # Find the symbol table for this file
                        if type_file_path in self.symbol_tables:
                            type_sym_table = self.symbol_tables[type_file_path]

                            # Look for the attribute in this class
                            class_scope = f"{type_file_path}::{class_name}"
                            obj_type = type_sym_table.get_variable_type(
                                attr_name, class_scope
                            )

                            if obj_type:
                                current_sym_table = type_sym_table
                                current_file = type_file_path
                    else:
                        # If simple type name, look in the current symbol table first
                        class_scope = f"{current_file}::{current_type}"
                        obj_type = current_sym_table.get_variable_type(
                            attr_name, class_scope
                        )

                        # If not found and we have a type name but no file info,
                        # we need to search for the class definition across files
                        if not obj_type:
                            for file_path, sym_table in self.symbol_tables.items():
                                if current_type in sym_table.classes:
                                    # Found the class definition
                                    class_scope = f"{file_path}::{current_type}"
                                    obj_type = sym_table.get_variable_type(
                                        attr_name, class_scope
                                    )

                                    if obj_type:
                                        current_sym_table = sym_table
                                        current_file = file_path
                                        break

                if obj_type:
                    # Successfully found the type of this attribute
                    current_type = obj_type
                    type_chain.append(current_type)
                    file_path_chain.append(current_file)

                    # Update scope for next attribute lookup
                    current_scope = f"{current_file}::{current_type}"

                    # If this is the last attribute in the chain, find the method
                    if i == len(attr_chain) - 1:
                        method_found = False

                        # First look in the current type's symbol table
                        if "::" in current_type:
                            # If fully qualified type (file::class)
                            parts = current_type.split("::")
                            type_file_path = parts[0]
                            class_name = parts[1]

                            if type_file_path in self.symbol_tables:
                                type_sym_table = self.symbol_tables[type_file_path]
                                method_path = type_sym_table.get_method(
                                    class_name, method_name
                                )

                                if method_path:
                                    logger.info(
                                        f"Found method on type chain: {type_chain} -> {method_path}"
                                    )
                                    self.graph.add_edge(
                                        caller, method_path, edge_type="method_call"
                                    )
                                    method_found = True
                        else:
                            # If simple type name, look in all symbol tables for this class
                            for _, sym_table in self.symbol_tables.items():
                                if current_type in sym_table.classes:
                                    method_path = sym_table.get_method(
                                        current_type, method_name
                                    )

                                    if method_path:
                                        logger.info(
                                            f"Found method on type: {current_type} -> {method_path}"
                                        )
                                        self.graph.add_edge(
                                            caller, method_path, edge_type="method_call"
                                        )
                                        method_found = True
                                        break

                        return method_found
                else:
                    # Couldn't find the type for this attribute
                    logger.debug(
                        f"Could not find type for attribute: {attr_name} in type: {current_type}"
                    )
                    return False

        # Handle other cases like module chains
        # ...

        return False


class ReferenceBuilder:
    def __init__(self, graph):
        self.graph = graph
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
                            logger.error(f"Syntax error in {file_path}, skipping...")
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

    # Now build the reference edges between functions using ReferenceBuilder
    reference_builder = ReferenceBuilder(graph)
    reference_builder.build_references(repo_path)

    # Print some statistics
    print(f"Total nodes in graph: {len(graph.nodes)}")
    print(f"Total edges in graph: {len(graph.edges)}")

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
