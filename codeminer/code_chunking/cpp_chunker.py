#!/usr/bin/env python3
"""
C++ specific code chunker implementation.
"""

from typing import List, Optional, Tuple

from .base import BaseCodeChunker


class CppCodeChunker(BaseCodeChunker):
    """Code chunker specifically for C++ files."""

    def __init__(
        self,
        max_lines_per_chunk: Optional[int] = None,
        chunk_depth: int = 2,
        enable_max_split: bool = True,
        include_header_epilogue: bool = False,
    ):
        """Initialize the C++ code chunker."""
        super().__init__(
            "cpp",
            max_lines_per_chunk=max_lines_per_chunk,
            chunk_depth=chunk_depth,
            enable_max_split=enable_max_split,
            include_header_epilogue=include_header_epilogue,
        )

    def _find_top_level_definitions(self, root_node) -> List[Tuple]:
        """
        Find all top-level function and class definitions in C++.

        Args:
            root_node: Root node of the AST

        Returns:
            List of tuples (node, name, type) for each top-level definition
        """
        definitions = []

        # For C++, look for function_definition and class_specifier
        for node in self._find_nodes_by_type(root_node, "function_definition"):
            name = self._extract_function_name(node)
            if name:
                definitions.append((node, name, "function"))

        for node in self._find_nodes_by_type(root_node, "class_specifier"):
            name = self._extract_class_name(node)
            if name:
                definitions.append((node, name, "class"))
                # Extract methods only if chunk_depth >= 2
                if self.chunk_depth >= 2:
                    methods = self._find_class_methods(node)
                    definitions.extend(methods)

        # Sort by start line
        definitions.sort(key=lambda x: x[0].start_point[0])
        return definitions

    def _extract_function_name(self, node) -> Optional[str]:
        """
        Extract function name from C++ function_definition node.

        Args:
            node: AST node representing a C++ function definition

        Returns:
            Function name or None if extraction failed
        """
        for child in node.children:
            if child.type == "function_declarator":
                for grandchild in child.children:
                    if grandchild.type in ("identifier", "field_identifier"):
                        return grandchild.text.decode("utf-8")
        return None

    def _extract_class_name(self, node) -> Optional[str]:
        """
        Extract class name from C++ class_specifier node.

        Args:
            node: AST node representing a C++ class definition

        Returns:
            Class name or None if extraction failed
        """
        for child in node.children:
            if child.type == "type_identifier":
                return child.text.decode("utf-8")
        return None

    def _find_class_methods(self, class_node) -> List[Tuple]:
        """
        Find all method definitions within a C++ class.

        Args:
            class_node: AST node representing a C++ class definition

        Returns:
            List of tuples (node, name, type) for each method definition
        """
        methods = []

        # Look for the class body (field_declaration_list)
        for child in class_node.children:
            if child.type == "field_declaration_list":
                # Within the class body, look for function definitions (methods)
                for member in child.children:
                    if member.type == "function_definition":
                        method_name = self._extract_method_name(member)
                        if method_name:
                            methods.append((member, method_name, "method"))
                    # Also look for function declarations that might be methods
                    elif member.type == "declaration":
                        # Check if this declaration contains a function declarator
                        for decl_child in member.children:
                            if decl_child.type == "function_declarator":
                                method_name = self._extract_method_name_from_declarator(
                                    decl_child
                                )
                                if method_name:
                                    methods.append((member, method_name, "method"))

        return methods

    def _extract_method_name(self, node) -> Optional[str]:
        """
        Extract method name from C++ function_definition node within a class.

        Args:
            node: AST node representing a C++ method definition

        Returns:
            Method name or None if extraction failed
        """
        # Method name extraction is the same as function name extraction
        return self._extract_function_name(node)

    def _extract_method_name_from_declarator(self, declarator_node) -> Optional[str]:
        """
        Extract method name from C++ function_declarator node.

        Args:
            declarator_node: AST node representing a C++ function declarator

        Returns:
            Method name or None if extraction failed
        """
        for child in declarator_node.children:
            if child.type in ("identifier", "field_identifier"):
                return child.text.decode("utf-8")
        return None
