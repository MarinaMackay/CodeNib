from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence


class AgentMode(str, Enum):
    """Execution mode hint for downstream operators."""

    AGENTLESS = "agentless"
    AGENT = "agent"


@dataclass(slots=True)
class QueryHints:
    """Lightweight bag of modifiers that refine logical operators."""

    predicate: Optional[str] = None
    fields: Optional[List[str]] = None
    normalization: Optional[str] = None
    engine: Optional[str] = None
    agent_mode: AgentMode = AgentMode.AGENTLESS
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LogicalOp:
    """Base class for logical operators."""

    hints: QueryHints = field(default_factory=QueryHints)

    def iter_children(self) -> Iterable["LogicalOp"]:
        """Yield child operators for tree traversals."""
        return ()

    def with_children(self, children: Sequence["LogicalOp"]) -> "LogicalOp":
        """Return a copy with updated children. Leaf operators simply validate."""
        if children:
            raise ValueError(
                f"{type(self).__name__} is a leaf and cannot accept children."
            )
        return self


@dataclass(slots=True)
class Search(LogicalOp):
    """Logical search over a view or index."""

    view: str = "default"
    k: int = 10


@dataclass(slots=True)
class Filter(LogicalOp):
    """Logical predicate filter."""

    predicate: str = ""
    source: LogicalOp = field(default_factory=LogicalOp)

    def iter_children(self) -> Iterable[LogicalOp]:
        yield self.source

    def with_children(self, children: Sequence[LogicalOp]) -> "Filter":
        if len(children) != 1:
            raise ValueError("Filter expects a single child.")
        return replace(self, source=children[0])


@dataclass(slots=True)
class Transform(LogicalOp):
    """Query or document transform (normalization, rewrite, rewriting agents)."""

    transform_id: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    source: LogicalOp = field(default_factory=LogicalOp)

    def iter_children(self) -> Iterable[LogicalOp]:
        yield self.source

    def with_children(self, children: Sequence[LogicalOp]) -> "Transform":
        if len(children) != 1:
            raise ValueError("Transform expects a single child.")
        return replace(self, source=children[0])


@dataclass(slots=True)
class Expand(LogicalOp):
    """Expansion step over a graph or related structure."""

    graph: str = ""
    spec: Dict[str, Any] = field(default_factory=dict)
    source: LogicalOp = field(default_factory=LogicalOp)

    def iter_children(self) -> Iterable[LogicalOp]:
        yield self.source

    def with_children(self, children: Sequence[LogicalOp]) -> "Expand":
        if len(children) != 1:
            raise ValueError("Expand expects a single child.")
        return replace(self, source=children[0])


@dataclass(slots=True)
class Rerank(LogicalOp):
    """Logical rerank stage."""

    model: str = ""
    top_m: int = 10
    source: LogicalOp = field(default_factory=LogicalOp)

    def iter_children(self) -> Iterable[LogicalOp]:
        yield self.source

    def with_children(self, children: Sequence[LogicalOp]) -> "Rerank":
        if len(children) != 1:
            raise ValueError("Rerank expects a single child.")
        return replace(self, source=children[0])


LogicalQuery = LogicalOp


@dataclass(slots=True)
class LogicalPlan:
    """Rooted logical plan used for the first IR layer."""

    root: LogicalQuery
    metadata: Dict[str, Any] = field(default_factory=dict)

    def walk(self) -> Iterable[LogicalOp]:
        """Depth-first walk of the logical plan."""
        stack: List[LogicalOp] = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(list(node.iter_children())))
