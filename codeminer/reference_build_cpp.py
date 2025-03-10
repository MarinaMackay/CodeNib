# reference_build_cpp.py
import os
import networkx as nx
import clang.cindex

from .treetransformer_cpp import SymbolTableBuilder

import clang.cindex
# location of llvm
clang.cindex.Config.set_library_file("/opt/homebrew/opt/llvm/lib/libclang.dylib")

class ReferenceVisitor:

    def __init__(self, 
                 graph, 
                 symbol_table, 
                 file_path):
        
        self.graph = graph
        self.symbol_table = symbol_table
        self.file_path = file_path
        self.current_function = None

    def visit(self, cursor):
        for node in cursor.get_children():
            # function definition
            if node.kind in (clang.cindex.CursorKind.FUNCTION_DECL, clang.cindex.CursorKind.CXX_METHOD):
                prev_function = self.current_function
                self.current_function = self.get_qualified_name(node)
                self.graph.add_node(self.current_function, type="function")
                self.visit(node)
                self.current_function = prev_function

            # call expression
            elif node.kind == clang.cindex.CursorKind.CALL_EXPR:
                callee = self.get_called_function(node)
                if self.current_function and callee:
                    self.graph.add_edge(self.current_function, callee, edge_type="calls")
                self.visit(node)
            else:
                self.visit(node)

    def get_qualified_name(self, cursor):
        names = []
        cur = cursor
        while cur is not None and cur.kind != clang.cindex.CursorKind.TRANSLATION_UNIT:
            if cur.spelling:
                names.insert(0, cur.spelling)
            cur = cur.semantic_parent
        return "::".join(names)

    def get_called_function(self, call_expr):
        if hasattr(call_expr, 'referenced') and call_expr.referenced:
            return self.get_qualified_name(call_expr.referenced)
        return call_expr.spelling

def build_references(repo_path):

    graph = nx.DiGraph()
    index = clang.cindex.Index.create()

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith((".cpp", ".cc", ".cxx", ".h", ".hpp")): # ?
                file_path = os.path.join(root, file)
                try:
                    tu = index.parse(file_path, args=["-std=c++17"])
                except Exception as e:
                    print(f"Failed to parse {file_path}: {e}")
                    continue

                builder = SymbolTableBuilder(file_path)
                builder.visit(tu.cursor)
                symbol_table = builder.symbol_table

                rel_file_path = os.path.relpath(file_path, repo_path)
                graph.add_node(rel_file_path, type="file")

                visitor = ReferenceVisitor(graph, symbol_table, rel_file_path)
                visitor.visit(tu.cursor)

    return graph

def build_graph(repo_path: str) -> nx.DiGraph:
    print(f"Analyzing C++ project at: {repo_path}")
    graph = build_references(repo_path)
    print(f"Total nodes: {len(graph.nodes)}")
    print(f"Total edges: {len(graph.edges)}")
    return graph
