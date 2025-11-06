from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Tuple

from ..ir_logical import AgentMode


@dataclass(slots=True)
class EngineCapability:
    """Capability descriptor for a specific engine or executor."""

    name: str
    supports_filter_pushdown: bool = False
    supports_topk: bool = True
    supports_hybrid: bool = False
    native_rerank: bool = False
    supports_fields: bool = False
    agent_modes: Tuple[AgentMode, ...] = (AgentMode.AGENTLESS,)
    extras: Dict[str, object] = field(default_factory=dict)

    def allows_agent(self, mode: AgentMode) -> bool:
        return mode in self.agent_modes


class EngineCapabilities(MutableMapping[str, EngineCapability]):
    """Registry style mapping for engine capability lookup."""

    def __init__(self, initial: Optional[Mapping[str, EngineCapability]] = None):
        self._caps: Dict[str, EngineCapability] = {}
        if initial:
            for key, value in initial.items():
                self._caps[key] = value

    def __getitem__(self, key: str) -> EngineCapability:
        return self._caps[key]

    def __setitem__(self, key: str, value: EngineCapability) -> None:
        self._caps[key] = value

    def __delitem__(self, key: str) -> None:
        del self._caps[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._caps)

    def __len__(self) -> int:
        return len(self._caps)

    def lookup(self, engine: str) -> Optional[EngineCapability]:
        return self._caps.get(engine)

    def values_for(self, *engines: str) -> Iterable[EngineCapability]:
        for engine in engines:
            cap = self.lookup(engine)
            if cap is not None:
                yield cap
