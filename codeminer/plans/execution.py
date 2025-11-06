from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional

from .ir_exec import ExecutionGraph, ExecutionNode


class ExecutionEngine:
    """Simple executor that walks the execution graph and invokes registered kernels."""

    def __init__(
        self,
        kernels: Optional[Mapping[str, Callable[[ExecutionNode, list], object]]] = None,
    ):
        self._kernels: Dict[str, Callable[[ExecutionNode, list], object]] = dict(
            kernels or {}
        )

    def register(
        self, operator: str, kernel: Callable[[ExecutionNode, list], object]
    ) -> None:
        self._kernels[operator] = kernel

    def execute(self, graph: ExecutionGraph) -> Dict[str, object]:
        state: Dict[str, object] = {}
        for node in graph.iter_topo():
            kernel = self._kernels.get(node.operator)
            if kernel is None:
                raise KeyError(f"No kernel registered for operator '{node.operator}'")
            inputs = [state[dep] for dep in node.deps]
            state[node.node_id] = kernel(node, inputs)
        return state
