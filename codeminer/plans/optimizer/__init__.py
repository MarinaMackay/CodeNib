"""
Optimizer package containing lowering pipeline, rewrite rules, capability tables,
and runtime feedback utilities.
"""

from .caps import EngineCapabilities, EngineCapability
from .costing import CostEstimate, CostModel
from .rewriter import LoweringPipeline
from .stats import PlanStatsStore

__all__ = [
    "LoweringPipeline",
    "EngineCapability",
    "EngineCapabilities",
    "CostEstimate",
    "CostModel",
    "PlanStatsStore",
]
