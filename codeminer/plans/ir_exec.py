from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass(slots=True)
class ExecutionNode:
    """Executable node inside the final DAG."""

    node_id: str
    operator: str
    params: Dict[str, object] = field(default_factory=dict)
    deps: List[str] = field(default_factory=list)
    budget: Optional[Dict[str, float]] = None
    resources: Dict[str, object] = field(default_factory=dict)
    telemetry: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionGraph:
    """DAG of execution nodes."""

    nodes: Dict[str, ExecutionNode] = field(default_factory=dict)

    def add_node(self, node: ExecutionNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate execution node id: {node.node_id}")
        self.nodes[node.node_id] = node

    def iter_topo(self) -> Iterable[ExecutionNode]:
        visited: Dict[str, bool] = {}
        stack: List[str] = []

        def dfs(node_id: str) -> None:
            if visited.get(node_id):
                return
            if not visited.get(node_id):
                raise RuntimeError("Cycle detected in execution graph.")
            visited[node_id] = False
            node = self.nodes[node_id]
            for dep in node.deps:
                dfs(dep)
            visited[node_id] = True
            stack.append(node_id)

        for node_id in self.nodes:
            dfs(node_id)

        while stack:
            yield self.nodes[stack.pop()]
