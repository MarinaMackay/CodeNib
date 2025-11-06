from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..ir_algebra import (
    AlgebraOp,
    AlgebraPlan,
    HybridUnion,
    RelationRef,
    Selection,
    TopK,
)
from ..ir_exec import ExecutionGraph, ExecutionNode
from ..ir_logical import LogicalPlan
from ..ir_physical import PhysicalOp, PhysicalOperator, PhysicalPlan
from .caps import EngineCapabilities
from .costing import CostModel
from .rules import (
    AlgebraRewriteRule,
    FilterPushdownRule,
    HybridFusionRule,
    LogicalToAlgebra,
)
from .stats import PlanStatsStore


@dataclass(slots=True)
class LoweringResult:
    logical: LogicalPlan
    algebra: AlgebraPlan
    physical: PhysicalPlan
    execution: ExecutionGraph


class LoweringPipeline:
    """Coordinates the successive lowering passes from logical to execution DAG."""

    def __init__(
        self,
        caps: Optional[EngineCapabilities] = None,
        cost_model: Optional[CostModel] = None,
        stats: Optional[PlanStatsStore] = None,
        algebra_rules: Optional[Sequence[AlgebraRewriteRule]] = None,
        logical_to_algebra: Optional[LogicalToAlgebra] = None,
    ) -> None:
        self.caps = caps or EngineCapabilities()
        self.cost_model = cost_model or CostModel()
        self.stats = stats or PlanStatsStore()
        self.logical_to_algebra = logical_to_algebra or LogicalToAlgebra(self.caps)
        self.algebra_rules: List[AlgebraRewriteRule] = list(
            algebra_rules or (FilterPushdownRule(), HybridFusionRule())
        )

    def lower(self, logical_plan: LogicalPlan) -> LoweringResult:
        """Lower a logical plan through all IR layers into an execution graph."""
        normalized = self._normalize(logical_plan)
        algebra_root = self.logical_to_algebra.apply(normalized)
        algebra_plan = AlgebraPlan(root=algebra_root)
        optimized_algebra = self._apply_algebra_rules(algebra_plan.root)
        physical_plan = self._materialize_physical(optimized_algebra)
        execution_graph = self._schedule_execution(physical_plan)
        return LoweringResult(
            logical=normalized,
            algebra=AlgebraPlan(root=optimized_algebra),
            physical=physical_plan,
            execution=execution_graph,
        )

    def _normalize(self, plan: LogicalPlan) -> LogicalPlan:
        # Placeholder for normalization passes (e.g., canonical field ordering)
        return plan

    def _apply_algebra_rules(self, root: AlgebraOp) -> AlgebraOp:
        def apply_recursive(node: AlgebraOp) -> AlgebraOp:
            children = [apply_recursive(child) for child in node.iter_children()]
            node = node.with_children(children)
            for rule in self.algebra_rules:
                if rule.matches(node):
                    node = rule.transform(node)
            return node

        return apply_recursive(root)

    def _materialize_physical(self, algebra_root: AlgebraOp) -> PhysicalPlan:
        def lower(node: AlgebraOp) -> PhysicalOp:
            if isinstance(node, Selection):
                child = lower(node.source)
                return PhysicalOp(
                    operator=PhysicalOperator.REGEX_FILTER,
                    params={"predicate": node.predicate},
                    children=[child],
                )
            if isinstance(node, TopK):
                child = lower(node.source)
                engine = node.annotations.get("engine", "sparse")
                operator = (
                    PhysicalOperator.BM25_TOPK
                    if engine == "sparse"
                    else PhysicalOperator.FAISS_SEARCH
                )
                return PhysicalOp(
                    operator=operator,
                    params={"k": node.k},
                    children=[child],
                )
            if isinstance(node, RelationRef):
                return PhysicalOp(
                    operator=PhysicalOperator.PASS_THROUGH,
                    params={"source": node.name},
                    children=[],
                )
            if isinstance(node, HybridUnion):
                children = [lower(child) for child in node.branches]
                return PhysicalOp(
                    operator=PhysicalOperator.HYBRID_SEARCH,
                    params={"weights": node.weights},
                    children=children,
                )
            children = [lower(child) for child in node.iter_children()]
            return PhysicalOp(
                operator=PhysicalOperator.PASS_THROUGH,
                params={"op": type(node).__name__},
                children=children,
            )

        physical_root = lower(algebra_root)
        return PhysicalPlan(root=physical_root)

    def _schedule_execution(self, plan: PhysicalPlan) -> ExecutionGraph:
        graph = ExecutionGraph()
        counter = 0

        def emit(node: PhysicalOp) -> str:
            nonlocal counter
            deps = [emit(child) for child in node.children]
            node_id = f"op_{counter}"
            counter += 1
            exec_node = ExecutionNode(
                node_id=node_id,
                operator=node.operator.value,
                params=node.params,
                deps=deps,
            )
            graph.add_node(exec_node)
            return node_id

        emit(plan.root)
        return graph
