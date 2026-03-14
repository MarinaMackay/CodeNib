from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from ..ir_algebra import AlgebraOp
from ..ir_physical import PhysicalOp


@dataclass(slots=True)
class CostEstimate:
    """Unified estimate representation across passes."""

    latency_ms: float
    compute_cost: float = 0.0
    selectivity: Optional[float] = None
    extra: Dict[str, float] = None

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}


class CostModel:
    """Cost model interface for both algebraic and physical planning."""

    def estimate_algebra(self, node: AlgebraOp) -> CostEstimate:
        """Return an estimate for an algebra node."""
        return CostEstimate(latency_ms=0.0)

    def estimate_physical(self, node: PhysicalOp) -> CostEstimate:
        """Return an estimate for a physical operator."""
        return CostEstimate(latency_ms=0.0)
