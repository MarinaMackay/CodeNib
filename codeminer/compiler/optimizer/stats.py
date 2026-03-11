from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, MutableMapping, Optional


@dataclass(slots=True)
class PlanStats:
    latency_ms: float
    selectivity: float
    metadata: Dict[str, float] = field(default_factory=dict)


class PlanStatsStore:
    """In-memory stats registry with optional persistence hooks."""

    def __init__(self) -> None:
        self._stats: MutableMapping[str, PlanStats] = {}

    def record(self, key: str, stats: PlanStats) -> None:
        self._stats[key] = stats

    def get(self, key: str) -> Optional[PlanStats]:
        return self._stats.get(key)

    def items(self) -> Iterable[tuple[str, PlanStats]]:
        return self._stats.items()
