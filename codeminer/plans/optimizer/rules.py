from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from ..ir_algebra import AlgebraOp, HybridUnion, RelationRef, Selection, TopK
from ..ir_logical import LogicalPlan, LogicalQuery, Search
from ..ir_physical import PhysicalOp, PhysicalOperator
from .caps import EngineCapabilities
from .costing import CostModel


@dataclass(slots=True)
class RewriteContext:
    logical_plan: LogicalPlan
    caps: EngineCapabilities
    cost_model: CostModel


class LogicalRule(Protocol):
    """Interface for logical → algebra rewrites."""

    def apply(self, plan: LogicalPlan) -> LogicalPlan: ...


class AlgebraRewriteRule(Protocol):
    """Interface for algebra-level rewrites."""

    def matches(self, node: AlgebraOp) -> bool: ...

    def transform(self, node: AlgebraOp) -> AlgebraOp: ...


class HybridFusionRule:
    """Example rule that fuses sparse and dense branches into a hybrid union."""

    def matches(self, node: AlgebraOp) -> bool:
        return isinstance(node, TopK) and isinstance(node.source, HybridUnion)

    def transform(self, node: AlgebraOp) -> AlgebraOp:
        return node  # Placeholder: real logic would adjust weights and push top-k


class FilterPushdownRule:
    """Placeholder for selection pushdown."""

    def matches(self, node: AlgebraOp) -> bool:
        return isinstance(node, Selection)

    def transform(self, node: AlgebraOp) -> AlgebraOp:
        return node


class LogicalToAlgebra:
    """Simplistic lowering pass from logical to algebraic form."""

    def __init__(self, caps: EngineCapabilities):
        self._caps = caps

    def apply(self, plan: LogicalPlan) -> AlgebraOp:
        root = plan.root
        return self._lower_node(root)

    def _lower_node(self, node: LogicalQuery) -> AlgebraOp:
        if isinstance(node, Search):
            base = RelationRef(name=node.view)
            if node.hints.fields:
                base = RelationRef(name=f"{node.view}:{','.join(node.hints.fields)}")
            current: AlgebraOp = base
            if node.hints.predicate:
                current = Selection(predicate=node.hints.predicate, source=current)
            return TopK(k=node.k, source=current)
        raise NotImplementedError(f"Lowering not implemented for {type(node).__name__}")


class PhysicalSelectionRule:
    """Placeholder that maps selection algebra nodes to physical filters."""

    def to_physical(self, node: AlgebraOp) -> Optional[PhysicalOp]:
        if isinstance(node, Selection):
            return PhysicalOp(
                operator=PhysicalOperator.REGEX_FILTER,
                params={"predicate": node.predicate},
            )
        return None
