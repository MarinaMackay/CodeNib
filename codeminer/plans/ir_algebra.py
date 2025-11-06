from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(slots=True)
class AlgebraOp:
    """Base algebra operator with optional annotations."""

    annotations: Dict[str, Any] = field(default_factory=dict)

    def iter_children(self) -> Iterable["AlgebraOp"]:
        return ()

    def with_children(self, children: Sequence["AlgebraOp"]) -> "AlgebraOp":
        if children:
            raise ValueError(f"{type(self).__name__} is a leaf operator.")
        return self

    def set_annotation(self, key: str, value: Any) -> "AlgebraOp":
        updated = replace(self)
        updated.annotations = dict(self.annotations)
        updated.annotations[key] = value
        return updated


@dataclass(slots=True)
class RelationRef(AlgebraOp):
    """Reference to an input relation (logical search seed)."""

    name: str = ""


@dataclass(slots=True)
class Selection(AlgebraOp):
    predicate: str = ""
    source: AlgebraOp = field(default_factory=RelationRef)

    def iter_children(self) -> Iterable[AlgebraOp]:
        yield self.source

    def with_children(self, children: Sequence[AlgebraOp]) -> "Selection":
        if len(children) != 1:
            raise ValueError("Selection expects one child.")
        return replace(self, source=children[0])


@dataclass(slots=True)
class Projection(AlgebraOp):
    fields: List[str] = field(default_factory=list)
    source: AlgebraOp = field(default_factory=RelationRef)

    def iter_children(self) -> Iterable[AlgebraOp]:
        yield self.source

    def with_children(self, children: Sequence[AlgebraOp]) -> "Projection":
        if len(children) != 1:
            raise ValueError("Projection expects one child.")
        return replace(self, source=children[0])


@dataclass(slots=True)
class TopK(AlgebraOp):
    k: int = 10
    source: AlgebraOp = field(default_factory=RelationRef)

    def iter_children(self) -> Iterable[AlgebraOp]:
        yield self.source

    def with_children(self, children: Sequence[AlgebraOp]) -> "TopK":
        if len(children) != 1:
            raise ValueError("TopK expects one child.")
        return replace(self, source=children[0])


@dataclass(slots=True)
class Sort(AlgebraOp):
    key: str = "score"
    descending: bool = True
    source: AlgebraOp = field(default_factory=RelationRef)

    def iter_children(self) -> Iterable[AlgebraOp]:
        yield self.source

    def with_children(self, children: Sequence[AlgebraOp]) -> "Sort":
        if len(children) != 1:
            raise ValueError("Sort expects one child.")
        return replace(self, source=children[0])


@dataclass(slots=True)
class Join(AlgebraOp):
    left: AlgebraOp = field(default_factory=RelationRef)
    right: AlgebraOp = field(default_factory=RelationRef)
    join_spec: Dict[str, Any] = field(default_factory=dict)

    def iter_children(self) -> Iterable[AlgebraOp]:
        yield self.left
        yield self.right

    def with_children(self, children: Sequence[AlgebraOp]) -> "Join":
        if len(children) != 2:
            raise ValueError("Join expects two children.")
        return replace(self, left=children[0], right=children[1])


@dataclass(slots=True)
class HybridUnion(AlgebraOp):
    """Represents hybrid (e.g., sparse + dense) fusion."""

    branches: List[AlgebraOp] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)

    def iter_children(self) -> Iterable[AlgebraOp]:
        yield from self.branches

    def with_children(self, children: Sequence[AlgebraOp]) -> "HybridUnion":
        return replace(self, branches=list(children))


@dataclass(slots=True)
class AlgebraPlan:
    """Normalized algebra plan ready for rule-based optimization."""

    root: AlgebraOp

    def walk(self) -> Iterable[AlgebraOp]:
        stack: List[AlgebraOp] = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(list(node.iter_children())))
