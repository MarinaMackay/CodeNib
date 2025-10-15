#!/usr/bin/env python3
"""
Python-specific code chunker implementation.
"""

from typing import List, Optional, Tuple

from .base import BaseCodeChunker


class PythonCodeChunker(BaseCodeChunker):
    """Code chunker specifically for Python files."""

    def __init__(
        self,
        max_lines_per_chunk: Optional[int] = 200,
        chunk_depth: int = 2,
        enable_max_split: bool = True,
    ):
        """Initialize the Python code chunker."""
        super().__init__(
            "python",
            max_lines_per_chunk=max_lines_per_chunk,
            chunk_depth=chunk_depth,
            enable_max_split=enable_max_split,
        )

    def _find_top_level_definitions(self, root_node) -> List[Tuple]:
        """
        Find all top-level function and class definitions in Python.
        Handles decorated_definition, function_definition, and class_definition nodes.

        Args:
            root_node: Root node of the AST

        Returns:
            List of tuples (node, name, type) for each top-level definition
        """
        definitions = []

        # For Python, look for function_definition, class_definition, and decorated_definition at module level
        for child in root_node.children:
            if child.type == "decorated_definition":
                # Extract the actual definition (function or class) from decorated_definition
                actual_def = self._extract_definition_from_decorated(child)
                if actual_def:
                    def_type = actual_def.type
                    if def_type in ("function_definition", "async_function_definition"):
                        name = self._extract_function_name(actual_def)
                        if name:
                            # Use the decorated_definition node (includes decorators)
                            definitions.append((child, name, "function"))
                    elif def_type == "class_definition":
                        name = self._extract_class_name(actual_def)
                        if name:
                            # Use the decorated_definition node (includes decorators)
                            definitions.append((child, name, "class"))
                            # Extract methods only if chunk_depth >= 2
                            if self.chunk_depth >= 2:
                                methods = self._find_class_methods(actual_def)
                                definitions.extend(methods)
            elif child.type in ("function_definition", "async_function_definition"):
                name = self._extract_function_name(child)
                if name:
                    definitions.append((child, name, "function"))
            elif child.type == "class_definition":
                name = self._extract_class_name(child)
                if name:
                    definitions.append((child, name, "class"))
                    # Extract methods only if chunk_depth >= 2
                    if self.chunk_depth >= 2:
                        methods = self._find_class_methods(child)
                        definitions.extend(methods)

        # Sort by start line
        definitions.sort(key=lambda x: x[0].start_point[0])
        return definitions

    def _extract_definition_from_decorated(self, decorated_node) -> Optional[object]:
        """
        Extract the actual definition node from a decorated_definition node.

        Args:
            decorated_node: AST node representing a decorated_definition

        Returns:
            The function_definition, async_function_definition, or class_definition node, or None
        """
        for child in decorated_node.children:
            if child.type in (
                "function_definition",
                "async_function_definition",
                "class_definition",
            ):
                return child
        return None

    def _extract_function_name(self, node) -> Optional[str]:
        """
        Extract function name from Python function_definition or async_function_definition node.

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
        Handles function_definition, async_function_definition, and decorated_definition nodes.

        Args:
            class_node: AST node representing a Python class definition

        Returns:
            List of tuples (node, name, type) for each method definition
        """
        methods = []

        class_name = self._extract_class_name(class_node)
        if not class_name:
            return methods

        # Look for the class body
        for child in class_node.children:
            if child.type == "block":
                # Within the class body, look for function definitions (methods)
                for stmt in child.children:
                    if stmt.type == "decorated_definition":
                        # Extract the actual method definition from decorated_definition
                        actual_def = self._extract_definition_from_decorated(stmt)
                        if actual_def and actual_def.type in (
                            "function_definition",
                            "async_function_definition",
                        ):
                            method_name = self._extract_method_name(actual_def)
                            if method_name:
                                # Include class name prefix for node_id
                                full_method_name = f"{class_name}.{method_name}"
                                # Use the decorated_definition node (includes decorators)
                                methods.append((stmt, full_method_name, "method"))
                    elif stmt.type in (
                        "function_definition",
                        "async_function_definition",
                    ):
                        method_name = self._extract_method_name(stmt)
                        if method_name:
                            # Include class name prefix for node_id
                            full_method_name = f"{class_name}.{method_name}"
                            methods.append((stmt, full_method_name, "method"))

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
