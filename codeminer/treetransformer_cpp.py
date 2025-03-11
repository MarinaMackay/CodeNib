# treetransformer_cpp.py
import clang.cindex

class SymbolTable:
    def __init__(self):
        self.classes = {}
        self.functions = {}
        self.variables = {}
        self.includes = set()

    def add_class(self, name, location):
        if name not in self.classes:
            self.classes[name] = {
                "location": location, 
                "methods": {}, 
                "attributes": set()
            }
        return self.classes[name]

    def add_function(self, name, location):
        self.functions[name] = location

    def add_method(self, class_name, method_name, location):
        if class_name in self.classes:
            self.classes[class_name]["methods"][method_name] = location

    def add_variable(self, scope, name, var_type):
        self.variables[(scope, name)] = var_type

    def add_include(self, header):
        self.includes.add(header)


class SymbolTableBuilder:
    def __init__(self, file_path):
        self.file_path = file_path
        self.symbol_table = SymbolTable()
        self.current_namespace = ""
        self.current_class = ""

    def get_qualified_name(self, name):
        parts = []
        if self.current_namespace:
            parts.append(self.current_namespace)
        if self.current_class:
            parts.append(self.current_class)
        parts.append(name)
        return "::".join(parts)

    def visit(self, cursor):
        for node in cursor.get_children():
            # namespace
            if node.kind == clang.cindex.CursorKind.NAMESPACE:
                prev_namespace = self.current_namespace
                if self.current_namespace:
                    self.current_namespace = self.current_namespace + "::" + node.spelling
                else:
                    self.current_namespace = node.spelling
                self.visit(node)
                self.current_namespace = prev_namespace

            # class
            elif node.kind in (clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL):
                prev_class = self.current_class
                qualified_class = self.get_qualified_name(node.spelling)
                self.current_class = qualified_class
                self.symbol_table.add_class(qualified_class, node.location)
                self.visit(node)
                self.current_class = prev_class

            # function
            elif node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                qualified_func = self.get_qualified_name(node.spelling)
                self.symbol_table.add_function(qualified_func, node.location)
                self.visit(node)

            # cpp method
            elif node.kind == clang.cindex.CursorKind.CXX_METHOD:
                qualified_method = self.get_qualified_name(node.spelling)
                if self.current_class:
                    self.symbol_table.add_method(self.current_class, node.spelling, node.location)
                self.visit(node)

            # variable
            elif node.kind == clang.cindex.CursorKind.VAR_DECL:
                var_type = node.type.spelling if node.type is not None else "unknown"
                scope = self.current_class if self.current_class else (self.current_namespace if self.current_namespace else "global")
                self.symbol_table.add_variable(scope, node.spelling, var_type)
                self.visit(node)

            # include
            elif node.kind == clang.cindex.CursorKind.INCLUSION_DIRECTIVE:
                self.symbol_table.add_include(node.spelling)
            else:
                self.visit(node)
