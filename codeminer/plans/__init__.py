"""
Intermediate representation layers and lowering utilities for Codeminer plans.

The package exposes four IR layers (logical, algebraic, physical, execution)
alongside optimization plumbing used to turn user intents into executable jobs.
"""

from .ir_algebra import AlgebraOp, AlgebraPlan
from .ir_exec import ExecutionGraph, ExecutionNode
from .ir_logical import (
    AgentMode,
    Expand,
    Filter,
    LogicalPlan,
    LogicalQuery,
    Rerank,
    Search,
    Transform,
)
from .ir_physical import PhysicalOp, PhysicalPlan

__all__ = [
    "AgentMode",
    "LogicalPlan",
    "LogicalQuery",
    "Search",
    "Filter",
    "Expand",
    "Transform",
    "Rerank",
    "AlgebraOp",
    "AlgebraPlan",
    "PhysicalOp",
    "PhysicalPlan",
    "ExecutionNode",
    "ExecutionGraph",
]
