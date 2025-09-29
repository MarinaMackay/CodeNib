#!/usr/bin/env python3
"""
Python-specific code chunker implementation.
"""

from typing import List, Optional, Tuple

from .base import BaseCodeChunker


class PythonCodeChunker(BaseCodeChunker):
    """Code chunker specifically for Python files."""

    def __init__(self, max_lines_per_chunk: Optional[int] = None):
        """Initialize the Python code chunker."""
        super().__init__("python", max_lines_per_chunk=max_lines_per_chunk)

    def _find_top_level_definitions(self, root_node) -> List[Tuple]:
        """
        Find all top-level function and class definitions in Python.

        Args:
            root_node: Root node of the AST

        Returns:
            List of tuples (node, name, type) for each top-level definition
        """
        definitions = []

        # For Python, look for function_definition and class_definition at module level
        for child in root_node.children:
            if child.type == "function_definition":
                name = self._extract_function_name(child)
                if name:
                    definitions.append((child, name, "function"))
            elif child.type == "class_definition":
                name = self._extract_class_name(child)
                if name:
                    definitions.append((child, name, "class"))
                    # Also extract methods within this class
                    methods = self._find_class_methods(child)
                    definitions.extend(methods)

        # Sort by start line
        definitions.sort(key=lambda x: x[0].start_point[0])
        return definitions

    def _extract_function_name(self, node) -> Optional[str]:
        """
        Extract function name from Python function_definition node.

        Args:
            node: AST node representing a Python function definition

        Returns:
            Function name or None if extraction failed
        """
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    def _extract_class_name(self, node) -> Optional[str]:
        """
        Extract class name from Python class_definition node.

        Args:
            node: AST node representing a Python class definition

        Returns:
            Class name or None if extraction failed
        """
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    def _find_class_methods(self, class_node) -> List[Tuple]:
        """
        Find all method definitions within a Python class.

        Args:
            class_node: AST node representing a Python class definition

        Returns:
            List of tuples (node, name, type) for each method definition
        """
        methods = []

        # Look for the class body
        for child in class_node.children:
            if child.type == "block":
                # Within the class body, look for function definitions (methods)
                for stmt in child.children:
                    if stmt.type == "function_definition":
                        method_name = self._extract_method_name(stmt)
                        if method_name:
                            methods.append((stmt, method_name, "method"))

        return methods

    def _extract_method_name(self, node) -> Optional[str]:
        """
        Extract method name from Python function_definition node within a class.

        Args:
            node: AST node representing a Python method definition

        Returns:
            Method name or None if extraction failed
        """
        # Method name extraction is the same as function name extraction
        return self._extract_function_name(node)
