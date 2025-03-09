import ast
from collections import namedtuple

# Keep your existing definitions
Loc = namedtuple("Loc", ["file_name", "node_name", "start_line", "end_line"])


class SymbolTable:
    def __init__(self):
        self.classes = {}  # name -> class info
        self.variables = {}  # name -> type info
        self.functions = {}  # name -> function info
        self.imports = {}  # name -> imported module/function/class
        
    def add_class(self, name, file_path, methods=None):
        """Add a class to the symbol table"""
        if name not in self.classes:
            self.classes[name] = {
                'file_path': file_path,
                'methods': methods or {},
                'attributes': set()
            }
        return self.classes[name]
        
    def add_method(self, class_name, method_name, file_path):
        """Add a method to a class"""
        if class_name in self.classes:
            self.classes[class_name]['methods'][method_name] = file_path
            
    def add_function(self, name, file_path):
        """Add a function to the symbol table"""
        self.functions[name] = file_path
        
    def add_variable(self, name, var_type, scope):
        """Add a variable with its type to the symbol table"""
        self.variables[(scope, name)] = var_type
        
    def add_attribute(self, class_name, attr_name):
        """Add an attribute to a class"""
        if class_name in self.classes:
            self.classes[class_name]['attributes'].add(attr_name)
            
    def add_import(self, name, import_path):
        """Add an import to the symbol table"""
        self.imports[name] = import_path
        
    def get_variable_type(self, name, scope):
        """Get the type of a variable in a specific scope"""
        if (scope, name) in self.variables:
            return self.variables[(scope, name)]
        return None
        
    def get_method(self, class_name, method_name):
        """Get a method from a class"""
        if class_name in self.classes and method_name in self.classes[class_name]['methods']:
            return self.classes[class_name]['methods'][method_name]
        return None
        
    def resolve_attribute_call(self, base_name, attr_name, scope, file_path):
        """Resolve a call like 'a.xxx()' where a is an instance of class A"""
        # First get the type of the base variable
        base_type = self.get_variable_type(base_name, scope)
        
        if not base_type:
            return None
        
        result = self.get_method(base_type, attr_name)

        # If not found and this is an imported class, we need to check
        # if the class is imported from another module
        if result is None and base_type in self.imports:
            # The class may be imported from another module
            imported_module = self.imports[base_type].split('.')[0]  # Get the module name
            print(f"DEBUG: Looking for imported class {base_type} in module {imported_module}")
            return f"my_project/{imported_module}.py::{base_type}::{attr_name}"
            
        # Now look for the method in that class
        return result

class SymbolTableBuilder(ast.NodeVisitor):
    """Build a symbol table for the entire codebase"""
    def __init__(self, file_path):
        self.file_path = file_path
        self.symbol_table = SymbolTable()
        self.current_class = None
        self.current_function = None
        self.current_scope = file_path
        
    def visit_ClassDef(self, node):
        class_name = node.name
        prev_class = self.current_class
        self.current_class = class_name
        
        # Add class to symbol table
        self.symbol_table.add_class(class_name, self.file_path)
        
        # Visit class body
        self.generic_visit(node)
        
        # Restore previous context
        self.current_class = prev_class
        
    def visit_FunctionDef(self, node):
        function_name = node.name
        prev_function = self.current_function
        
        if self.current_class:
            # This is a method
            self.current_function = f"{self.current_class}::{function_name}"
            self.current_scope = f"{self.file_path}::{self.current_class}::{function_name}"
            
            # Add method to class in symbol table
            self.symbol_table.add_method(self.current_class, function_name, 
                                        f"{self.file_path}::{self.current_class}::{function_name}")
        else:
            # This is a function
            self.current_function = function_name
            self.current_scope = f"{self.file_path}::{function_name}"
            
            # Add function to symbol table
            self.symbol_table.add_function(function_name, f"{self.file_path}::{function_name}")
        
        # Visit function body
        self.generic_visit(node)
        
        # Restore previous context
        self.current_function = prev_function
        self.current_scope = self.file_path if not prev_function else \
                           (f"{self.file_path}::{prev_function}" if not self.current_class else 
                            f"{self.file_path}::{self.current_class}::{prev_function}")
        
    def visit_Assign(self, node):
        # Only handle assignments with a single target for simplicity
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            
            # Try to determine variable type
            var_type = None
            
            # Case: a = Class()
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                var_type = node.value.func.id
                
            # Add variable type to symbol table
            if var_type:
                self.symbol_table.add_variable(var_name, var_type, self.current_scope)
                
        # Handle attribute assignments (self.x = y)
        elif len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute) and \
             isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id == 'self' and \
             self.current_class:
            
            attr_name = node.targets[0].attr
            # Add attribute to class
            self.symbol_table.add_attribute(self.current_class, attr_name)
            
        self.generic_visit(node)
        
    def visit_Import(self, node):
        """Track imported modules and their aliases."""
        for alias in node.names:
            imported_name = alias.name  # Full module name
            as_name = alias.asname or imported_name
            self.symbol_table.add_import(as_name, imported_name)
            
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Track specific imports from a module."""
        module = node.module or ""
        for alias in node.names:
            imported_name = f"{module}.{alias.name}"
            as_name = alias.asname or alias.name
            self.symbol_table.add_import(as_name, imported_name)
            
        self.generic_visit(node)
