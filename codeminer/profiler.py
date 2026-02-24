from __future__ import annotations

import logging
import threading
import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field, replace
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

from .log_utils import get_logger

__all__ = ["Profiler", "RangeStats"]


@dataclass
class RangeStats:
    """Aggregate timing statistics captured for a profiler section."""

    total: float = 0.0
    count: int = 0
    max_duration: float = 0.0
    min_duration: float = field(default_factory=lambda: float("inf"))
    errors: int = 0

    def update(self, duration: float, had_error: bool) -> None:
        self.total += duration
        self.count += 1
        if duration > self.max_duration:
            self.max_duration = duration
        if duration < self.min_duration:
            self.min_duration = duration
        if had_error:
            self.errors += 1

    @property
    def average(self) -> float:
        if not self.count:
            return 0.0
        return self.total / self.count

    @property
    def safe_min(self) -> float:
        if self.min_duration == float("inf"):
            return 0.0
        return self.min_duration


class Profiler:
    """
    Lightweight hierarchical profiler for logging execution time of code sections.

    The profiler is designed to replace hand-written timing calls and provide
    reusable instrumentation with per-section summaries. Sections can be used
    as context managers or decorators:

        profiler = Profiler("scip_indexer")

        with profiler.section("generate_index"):
            ...

        @profiler.time_function()
        def compute():
            ...
    """

    def __init__(
        self,
        name: str = "profiler",
        *,
        enabled: bool = True,
        logger: Optional[logging.Logger] = None,
        emit_events: bool = True,
        range_level: int = logging.DEBUG,
        summary_level: int = logging.INFO,
        indent: int = 2,
    ) -> None:
        self.name = name
        self.enabled = enabled
        self.emit_events = emit_events
        self.range_level = range_level
        self.summary_level = summary_level
        self.indent = indent
        self.logger = logger or get_logger(name)

        self._thread_state = threading.local()
        self._lock = threading.Lock()
        self._stats: Dict[str, RangeStats] = {}

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def section(
        self,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "_ProfilerRange":
        """
        Create a profiling section context manager for the given label.

        Args:
            label: Name of the section to record.
            metadata: Optional values to include in the emitted logs.
        """
        return _ProfilerRange(self, label, metadata)

    def time_function(
        self,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator helper for profiling function execution time.

        Args:
            label: Override label for the recorded section. Defaults to the
                function's qualified name.
            metadata: Optional metadata to include in the emitted logs.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            section_label = label or func.__qualname__

            @wraps(func)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                with self.section(section_label, metadata):
                    return func(*args, **kwargs)

            return wrapped

        return decorator

    def report(
        self, *, reset: bool = False, top_n: Optional[int] = None
    ) -> List[Tuple[str, RangeStats]]:
        """
        Emit an aggregated timing report ordered by total time spent per section.

        Args:
            reset: Clear stored statistics after reporting.
            top_n: Limit output to the N most expensive sections.
        Returns:
            List of (label, RangeStats snapshot) pairs sorted by total duration.
        """
        if not self.enabled:
            return []

        with self._lock:
            items = list(self._stats.items())

        if not items:
            self.logger.log(
                self.summary_level, "[%s] profiler: no sections recorded", self.name
            )
            return []

        total_sections = len(items)
        grand_total = max((stats.total for _, stats in items), default=0.0)

        # Order by total duration (descending) to mirror flame graph style reports.
        items.sort(key=lambda item: item[1].total, reverse=True)
        if top_n is not None:
            items = items[:top_n]

        summary_items: List[Tuple[str, RangeStats]] = []

        self.logger.log(
            self.summary_level,
            "[%s] profiler summary: %.3fs total across %d section(s)",
            self.name,
            grand_total,
            total_sections,
        )

        for label, stats in items:
            snapshot = replace(stats)
            summary_items.append((label, snapshot))
            extra = f", errors={stats.errors}" if stats.errors else ""
            self.logger.log(
                self.summary_level,
                "  %-40s total=%6.3fs count=%3d avg=%6.3fs min=%6.3fs max=%6.3fs%s",
                label,
                stats.total,
                stats.count,
                stats.average,
                stats.safe_min,
                stats.max_duration,
                extra,
            )

        if reset:
            self.reset()

        return summary_items

    def reset(self) -> None:
        """Clear collected statistics and reset the section stack."""
        with self._lock:
            self._stats.clear()
        # Reset per-thread stacks.
        self._thread_state.__dict__.clear()

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _record(
        self,
        label: str,
        duration: float,
        had_error: bool,
    ) -> None:
        with self._lock:
            stats = self._stats.get(label)
            if stats is None:
                stats = RangeStats()
                self._stats[label] = stats
            stats.update(duration, had_error)

    def _get_stack(self) -> list:
        stack = getattr(self._thread_state, "stack", None)
        if stack is None:
            stack = []
            self._thread_state.stack = stack
        return stack

    def _format_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        if not metadata:
            return ""
        if isinstance(metadata, dict):
            parts = [f"{key}={value}" for key, value in metadata.items()]
            return " (" + ", ".join(parts) + ")"
        return f" ({metadata})"

    def _indent(self, depth: int) -> str:
        return " " * max(depth * self.indent, 0)

    def _log_range_start(
        self,
        label: str,
        depth: int,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        if not (self.enabled and self.emit_events):
            return

        self.logger.log(
            self.range_level,
            "%s[start] %s%s",
            self._indent(depth),
            label,
            self._format_metadata(metadata),
        )

    def _log_range_stop(
        self,
        label: str,
        depth: int,
        duration: float,
        had_error: bool,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        if not (self.enabled and self.emit_events):
            return

        status = "error" if had_error else "stop"
        self.logger.log(
            self.range_level,
            "%s[%s] %s%s (%.3fs)",
            self._indent(depth),
            status,
            label,
            self._format_metadata(metadata),
            duration,
        )

    @staticmethod
    def _now() -> float:
        return time.perf_counter()


class _ProfilerRange(ContextDecorator):
    """Context manager used by Profiler.section."""

    __slots__ = (
        "_profiler",
        "label",
        "metadata",
        "start_time",
        "duration",
        "depth",
        "had_error",
    )

    def __init__(
        self,
        profiler: Profiler,
        label: str,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        self._profiler = profiler
        self.label = label
        self.metadata = metadata
        self.start_time: float = 0.0
        self.duration: float = 0.0
        self.depth: int = 0
        self.had_error: bool = False

    def __enter__(self) -> "_ProfilerRange":
        if not self._profiler.enabled:
            self.duration = 0.0
            self.depth = 0
            return self

        stack = self._profiler._get_stack()
        self.depth = len(stack)
        stack.append(self)

        self.start_time = self._profiler._now()
        self._profiler._log_range_start(self.label, self.depth, self.metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self._profiler.enabled:
            return False

        end_time = self._profiler._now()
        self.duration = end_time - self.start_time
        self.had_error = exc_type is not None

        stack = self._profiler._get_stack()
        if stack and stack[-1] is self:
            stack.pop()
        else:
            # Stack corruption should be rare, but guard against it so that
            # future profiling calls are not affected.
            try:
                stack.remove(self)
            except ValueError:
                stack.clear()

        self._profiler._record(self.label, self.duration, self.had_error)
        self._profiler._log_range_stop(
            self.label,
            self.depth,
            self.duration,
            self.had_error,
            self.metadata,
        )
        # Propagate exceptions to the caller.
        return False
