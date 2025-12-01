#!/usr/bin/env python3
"""
Rust-specific code chunker implementation.
"""

from typing import List, Optional, Tuple

from .base import BaseCodeChunker


class RustCodeChunker(BaseCodeChunker):
    """
    Code chunker for Rust source files.

    L1 entities: free functions plus struct/enum/trait/impl blocks.
    L2 entities: functions inside impl blocks.
    Skeleton mode: file skeleton lists L1 declarations; impl skeleton lists method signatures.
    """

    def __init__(
        self,
        max_lines_per_chunk: Optional[int] = 200,
        chunk_depth: int = 2,
        l2_level_exclusive: bool = True,
        **kwargs,
    ):
        super().__init__(
            "rust",
            max_lines_per_chunk=max_lines_per_chunk,
            chunk_depth=chunk_depth,
            l2_level_exclusive=l2_level_exclusive,
            **kwargs,
        )

    def _find_top_level_definitions(self, root_node) -> List[Tuple]:
        definitions: List[Tuple] = []

        include_type_level = self.chunk_depth < 2 or not self.l2_level_exclusive

        for child in root_node.children:
            if child.type == "function_item":
                name = self._extract_function_name(child)
                if name:
                    definitions.append((child, name, "function"))
            elif child.type in ("struct_item", "enum_item", "trait_item"):
                name = self._extract_type_name(child)
                if name and include_type_level:
                    chunk_type = child.type.replace("_item", "")
                    definitions.append((child, name, chunk_type))
            elif child.type == "impl_item":
                impl_label, target_name = self._extract_impl_label(child)
                if include_type_level:
                    definitions.append((child, impl_label, "impl"))
                if self.chunk_depth >= 2:
                    definitions.extend(self._find_impl_methods(child, target_name))

        definitions.sort(key=lambda entry: entry[0].start_point[0])
        return definitions

    def _extract_function_name(self, node) -> Optional[str]:
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    def _extract_type_name(self, node) -> Optional[str]:
        for child in node.children:
            if child.type in ("type_identifier", "identifier"):
                return child.text.decode("utf-8")
        return None

    def _extract_impl_label(self, node) -> Tuple[str, Optional[str]]:
        target_name = None
        for child in node.children:
            if child.type in (
                "type_identifier",
                "identifier",
                "scoped_identifier",
                "scoped_type_identifier",
            ):
                target_name = child.text.decode("utf-8")
                break
        label = f"impl {target_name}" if target_name else "impl"
        return label, target_name

    def _find_impl_methods(self, impl_node, target_name: Optional[str]) -> List[Tuple]:
        methods: List[Tuple] = []
        for child in impl_node.children:
            if child.type == "declaration_list":
                for decl in child.children:
                    if decl.type == "function_item":
                        method_name = self._extract_function_name(decl)
                        if method_name:
                            if target_name:
                                full_name = f"{target_name}::{method_name}"
                            else:
                                full_name = method_name
                            methods.append((decl, full_name, "method"))
        return methods

    def _extract_class_name(self, node) -> Optional[str]:
        # Rust does not have traditional classes; reuse struct/enum name extraction.
        return self._extract_type_name(node)

    def _extract_method_name(self, node) -> Optional[str]:
        return self._extract_function_name(node)

    def _get_child_definitions(self, node, def_type: str):
        """Return impl methods for impl skeletons."""
        if def_type != "impl":
            return []
        _, target_name = self._extract_impl_label(node)
        return self._find_impl_methods(node, target_name)
