import ast
import os
import sys
from collections import namedtuple

from .log_utils import get_logger

logger = get_logger(__name__)

# Keep your existing definitions
Loc = namedtuple("Loc", ["file_name", "node_name", "start_line", "end_line"])


def is_local_module(module_name: str, repo_path: str) -> bool:
    """
    Determine if a module is local to the repository (not a standard library or external package)

    Args:
        module_name: The module name (e.g., 'utils.helpers', 'os', 'numpy')
        repo_path: The repository root path

    Returns:
        bool: True if the module is local to the repository, False otherwise
    """
    # First check standard library modules - these are definitely not local

    if module_name in sys.stdlib_module_names:
        return False

    # Check if module exists as a directory or file in the repo
    base_module = module_name.split(".")[0]  # Get the root module name

    # Check if it's a directory package
    # logger.debug(f"Checking if {base_module} is a local module in {repo_path}")
    if os.path.isdir(os.path.join(repo_path, base_module)):
        return True

    # Check if it's a .py file
    if os.path.isfile(os.path.join(repo_path, f"{base_module}.py")):
        return True

    # If we get here, it's likely an external package
    return False


class SymbolTable:
    def __init__(self):
        self.classes = {}  # name -> class info
        self.variables = {}  # name -> type info
        self.functions = {}  # name -> function info
        self.imports = {}  # name -> imported module/function/class

    def add_class(self, name: str, file_path: str, methods=None):
        """Add a class to the symbol table"""
        if name not in self.classes:
            self.classes[name] = {
                "file_path": file_path,
                "methods": methods or {},
                "attributes": set(),
            }
        return self.classes[name]

    def add_method(self, class_name: str, method_name: str, file_path: str):
        """Add a method to a class"""
        if class_name in self.classes:
            self.classes[class_name]["methods"][method_name] = file_path

    def add_function(self, name: str, file_path: str):
        """Add a function to the symbol table"""
        self.functions[name] = file_path

    def add_variable(self, name: str, var_type: str, scope: str):
        """Add a variable with its type to the symbol table"""
        self.variables[(scope, name)] = var_type

    def add_attribute(self, class_name: str, attr_name: str):
        """Add an attribute to a class"""
        if class_name in self.classes:
            self.classes[class_name]["attributes"].add(attr_name)

    def add_import(
        self, name: str, import_path: str, is_local: bool = True, is_alias: bool = False
    ):
        """
        Add an import to the symbol table

        Args:
            name: The local name used for the import
            import_path: The full import path
            is_local: Whether this is a local module (not stdlib or external)
            is_alias: Whether this is an alias for the import
        """
        self.imports[name] = {
            "path": import_path,
            "is_local": is_local,
            "is_alias": is_alias,
        }

    def get_import(self, name: str) -> str | None:
        """Get information about an imported name"""
        if name in self.imports:
            return self.imports[name]
        return None

    def get_variable_type(self, name: str, scope: str) -> str | None:
        """Get the type of a variable in a specific scope"""
        if (scope, name) in self.variables:
            return self.variables[(scope, name)]
        return None

    def get_method(self, class_name: str, method_name: str) -> str | None:
        """Get a method from a class"""
        if (
            class_name in self.classes
            and method_name in self.classes[class_name]["methods"]
        ):
            return self.classes[class_name]["methods"][method_name]
        return None

    def resolve_attribute_call(
        self, base_name: str, attr_name: str, scope: str, symbol_tables: dict
    ) -> str | None:
        """Resolve a call like 'a.xxx()' where a is an instance of class A"""
        # First get the type of the base variable
        base_type = self.get_variable_type(base_name, scope)

        if not base_type:
            # Check if base_name itself is an imported module alias
            if base_name in self.imports:
                import_info = self.imports[base_name]

                # Skip external imports
                if not import_info.get("is_local", True):
                    logger.debug(
                        f"Skipping external module reference: {base_name}.{attr_name}"
                    )
                    return None

                imported_module = import_info["path"]
                # check if the import module using alias
                is_alias = import_info["is_alias"]
                # Handle the case where base_name is an alias for an imported module
                if is_alias:
                    # Handle module.Class() pattern, import a.AA as module
                    # a.AA is actually a path (imported_module)
                    # base_name should be dropped, since base_name == imported_module
                    # TODO: this seems to be more like refercne call?
                    module_parts = imported_module.split(".")
                    if len(module_parts) > 1:
                        # join all parts
                        module_path = "/".join(module_parts[:])
                        logger.info(
                            f"module_path: {module_path}, base_name: {base_name}, attr_name: {attr_name}"
                        )
                        return f"{module_path}.py::{attr_name}"
                    return f"{imported_module}.py::{attr_name}"
                else:
                    module_parts = imported_module.split(".")
                    if len(module_parts) > 1:
                        module_path = "/".join(module_parts[:-1])
                        logger.info(
                            f"Nested module detected: {module_parts}, base_name: {base_name}, attr_name: {attr_name}"
                        )
                        return f"{module_path}.py::{base_name}::{attr_name}"
                    else:
                        return f"{imported_module}.py::{base_name}::{attr_name}"
            return None

        # Check if this is a direct method call on a known class in the symbol table
        result = self.get_method(base_type, attr_name)
        if result:
            return result

        # Handle base_type being an imported module
        if base_type in self.imports:
            import_info = self.imports[base_type]

            # Skip external imports
            if not import_info.get("is_local", True):
                logger.debug(f"Skipping external class method: {base_type}.{attr_name}")
                return None

            imported_path = import_info["path"]
            # Handle nested modules like alg.alg.Alg
            module_parts = imported_path.split(".")
            # Create proper file path from module parts
            if len(module_parts) > 1:
                # Handle case of nested modules
                module_path = "/".join(module_parts[:-1])
                class_name = module_parts[-1]
                return f"{module_path}.py::{class_name}::{attr_name}"
            else:
                # Simple import case
                return f"{imported_path}.py::{base_type}::{attr_name}"

        return None


class SymbolTableBuilder(ast.NodeVisitor):
    """Build a symbol table for the entire codebase"""

    def __init__(self, file_path: str, repo_path: str = None):
        self.file_path = file_path
        self.symbol_table = SymbolTable()
        self.current_class = None
        self.current_function = None
        self.current_scope = file_path
        self.repo_path = repo_path

    def visit_ClassDef(self, node: ast.ClassDef):
        class_name = node.name
        prev_class = self.current_class
        self.current_class = class_name

        # Add class to symbol table
        self.symbol_table.add_class(class_name, self.file_path)

        # Visit class body
        self.generic_visit(node)

        # Restore previous context
        self.current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        function_name = node.name
        prev_function = self.current_function
        prev_scope = self.current_scope

        if self.current_class:
            # This is a method
            self.current_function = f"{self.current_class}::{function_name}"
            self.current_scope = (
                f"{self.file_path}::{self.current_class}::{function_name}"
            )

            # Add method to class in symbol table
            self.symbol_table.add_method(
                self.current_class,
                function_name,
                f"{self.file_path}::{self.current_class}::{function_name}",
            )
        else:
            # This is a function
            self.current_function = function_name
            self.current_scope = f"{self.file_path}::{function_name}"

            # Add function to symbol table
            self.symbol_table.add_function(
                function_name, f"{self.file_path}::{function_name}"
            )

        # Visit function body
        self.generic_visit(node)

        # Restore previous context
        self.current_function = prev_function
        self.current_scope = prev_scope

    def visit_Assign(self, node: ast.Assign):
        # Only handle assignments with a single target for simplicity
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id

            # Try to determine variable type
            var_type = None

            # Case: a = Class()
            if isinstance(node.value, ast.Call) and isinstance(
                node.value.func, ast.Name
            ):
                var_type = node.value.func.id

            # Add variable type to symbol table
            if var_type:
                self.symbol_table.add_variable(var_name, var_type, self.current_scope)
            # Case: alg = AAA.Alg() - handling module attributes
            elif isinstance(node.value, ast.Call) and isinstance(
                node.value.func, ast.Attribute
            ):
                if isinstance(node.value.func.value, ast.Name):
                    module_name = node.value.func.value.id
                    class_name = node.value.func.attr

                    # If the module is imported, track the variable as the class from that module
                    if module_name in self.symbol_table.imports:
                        imported_module = self.symbol_table.imports[module_name]
                        var_type = class_name
                        # Store the fully qualified class name for this variable
                        self.symbol_table.add_variable(
                            var_name,
                            f"{imported_module['path']}::{var_type}",
                            self.current_scope,
                        )

            # Add variable type to symbol table
            if var_type:
                self.symbol_table.add_variable(var_name, var_type, self.current_scope)

        # Handle attribute assignments (self.x = y)
        elif (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "self"
            and self.current_class
        ):

            attr_name = node.targets[0].attr
            # Add attribute to class
            self.symbol_table.add_attribute(self.current_class, attr_name)

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        """Track imported modules and their aliases."""
        for alias in node.names:
            imported_name = alias.name  # Full module name
            as_name = alias.asname or imported_name

            # Add flag to indicate if this is a local import
            is_local = is_local_module(imported_name, self.repo_path)
            is_alias = alias.asname is not None
            self.symbol_table.add_import(as_name, imported_name, is_local, is_alias)
            logger.debug(
                f"Importing {imported_name} as {as_name}, is_local: {is_local}, is_alias: {is_alias}"
            )

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.Import):
        """Track specific imports from a module."""
        module = node.module or ""

        # Check if this is a local import
        is_local = is_local_module(module, self.repo_path)

        for alias in node.names:
            imported_name = f"{module}.{alias.name}"
            as_name = alias.asname or alias.name
            is_alias = alias.asname is not None
            # Add flag to indicate if this is a local import
            self.symbol_table.add_import(as_name, imported_name, is_local, is_alias)
            # logger.debug(f"Importing {imported_name} as {as_name}, is_local: {is_local}, is_alias: {is_alias}")

        self.generic_visit(node)
