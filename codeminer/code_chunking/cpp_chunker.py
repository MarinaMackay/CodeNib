#!/usr/bin/env python3
"""
C++ specific code chunker implementation.
"""

from typing import List, Optional, Tuple

from .base import BaseCodeChunker


class CppCodeChunker(BaseCodeChunker):
    """Code chunker specifically for C++ files."""

    def __init__(self, max_lines_per_chunk: Optional[int] = None):
        """Initialize the C++ code chunker."""
        super().__init__("cpp", max_lines_per_chunk=max_lines_per_chunk)

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
