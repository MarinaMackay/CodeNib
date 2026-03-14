"""
Async DAG scheduler that executes a compiled ``ExecutionPlan``.

Features:
- Stage-based parallel execution via ``asyncio.gather``
- Checkpoint / partial rollback
- Bridge layer that maps skills → ``ExecutionNode`` → existing kernels

This module stays in Python even after the Rust migration — the Rust compiler
only *produces* the plan; Python *executes* it.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...agent.skills.core import SkillMetadata
from ...agent.skills.registry import SkillRegistry
from ..execution import ExecutionEngine
from ..ir_exec import ExecutionNode
from ..params import SessionContext, SessionParamRules, resolve_params
from ..resources import ResourcePlan
from .dependency_resolver import SkillDAG
from .policy import ExecutionPlan

# ---------------------------------------------------------------------------
# Result / checkpoint data
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SkillExecutionResult:
    """Outcome of running a single skill."""

    skill_id: str
    output: Any = None
    status: str = "pending"  # "success" | "failed" | "skipped"
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass(slots=True)
class CheckpointState:
    """Snapshot taken before a stage executes, for rollback."""

    stage_id: int
    completed_results: Dict[str, SkillExecutionResult] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class SkillScheduler:
    """
    Execute an ``ExecutionPlan`` asynchronously.

    The scheduler walks through stages, running each skill either via its
    ``executor_fn`` (if present) or by mapping it to an ``ExecutionNode`` and
    delegating to the existing ``ExecutionEngine`` kernel system.
    """

    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        engine: Optional[ExecutionEngine] = None,
        *,
        session_ctx: Optional[SessionContext] = None,
        session_rules: Optional[SessionParamRules] = None,
    ) -> None:
        self.registry = registry or SkillRegistry()
        self.engine = engine or ExecutionEngine()
        self.session_ctx = session_ctx
        self.session_rules = session_rules
        self.checkpoints: List[CheckpointState] = []

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        plan: ExecutionPlan,
        *,
        resource_plan: Optional[ResourcePlan] = None,
        enable_checkpoints: bool = True,
    ) -> Dict[str, Any]:
        """
        Run all stages in the plan, returning a dict mapping
        ``skill_id → output``.

        If a *resource_plan* is provided, the scheduler checks for
        blocking builds before starting.  Non-blocking staleness
        warnings are collected but execution proceeds.
        """
        if resource_plan and not resource_plan.can_execute:
            raise RuntimeError(
                f"Cannot execute: missing required indexes "
                f"{resource_plan.blocking_builds}. "
                f"Build them first or provide an IndexBuilderRegistry."
            )

        results: Dict[str, SkillExecutionResult] = {}
        state: Dict[str, Any] = {}

        for stage in plan.stages:
            if enable_checkpoints:
                self.checkpoints.append(
                    CheckpointState(
                        stage_id=stage.stage_id,
                        completed_results=dict(results),
                        state=dict(state),
                    )
                )

            stage_results = await self._execute_stage(stage.skill_ids, plan.dag, state)
            results.update(stage_results)

            for sid, res in stage_results.items():
                if res.status == "success":
                    state[sid] = res.output

        return state

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _execute_stage(
        self,
        skill_ids: List[str],
        dag: Optional[SkillDAG],
        state: Dict[str, Any],
    ) -> Dict[str, SkillExecutionResult]:
        tasks = [self._execute_one(sid, dag, state) for sid in skill_ids]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        results: Dict[str, SkillExecutionResult] = {}
        for sid, outcome in zip(skill_ids, outcomes, strict=False):
            if isinstance(outcome, BaseException):
                results[sid] = SkillExecutionResult(
                    skill_id=sid,
                    status="failed",
                    error=str(outcome),
                )
            else:
                results[sid] = outcome
        return results

    async def _execute_one(
        self,
        skill_id: str,
        dag: Optional[SkillDAG],
        state: Dict[str, Any],
    ) -> SkillExecutionResult:
        metadata = self.registry.get(skill_id)
        if metadata is None:
            return SkillExecutionResult(
                skill_id=skill_id,
                status="failed",
                error=f"Skill {skill_id!r} not in registry",
            )

        inputs = dag.nodes[skill_id].inputs if dag and skill_id in dag.nodes else {}
        start = time.monotonic()

        try:
            output = await self._run_skill(metadata, inputs, state)
            elapsed = (time.monotonic() - start) * 1000
            return SkillExecutionResult(
                skill_id=skill_id,
                output=output,
                status="success",
                duration_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            return SkillExecutionResult(
                skill_id=skill_id,
                status="failed",
                error=str(exc),
                duration_ms=elapsed,
            )

    async def _run_skill(
        self,
        metadata: SkillMetadata,
        inputs: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Any:
        """
        Execute a single skill.

        Strategy (in priority order):
        1. If the metadata has an ``executor_fn`` that does NOT raise
           ``NotImplementedError``, call it directly.
        2. Otherwise, map to an ``ExecutionNode`` and delegate to the
           existing ``ExecutionEngine`` kernel system.

        Only the parameters declared in the skill's input spec are forwarded
        to ``executor_fn``; extra params (e.g. ``return_content``) are
        kernel-level concerns and go to the engine path.
        """
        fn = metadata.executor_fn
        if fn is not None:
            # Filter inputs to only the declared skill parameters so that
            # kernel-level params don't cause TypeError on executor_fn.
            declared = {spec.name for spec in metadata.inputs}
            fn_inputs = {k: v for k, v in inputs.items() if k in declared}
            try:
                if metadata.async_capable and asyncio.iscoroutinefunction(fn):
                    return await fn(**fn_inputs)
                return await asyncio.get_event_loop().run_in_executor(
                    None, lambda: fn(**fn_inputs)
                )
            except NotImplementedError:
                pass  # fall through to engine path

        # Bridge to existing kernel system (receives ALL inputs as params).
        return await self._run_via_engine(metadata, inputs, state)

    def _resolve_skill_params(
        self,
        metadata: SkillMetadata,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge defaults → session → query-time inputs via layered resolution."""
        if not metadata.defaults:
            return dict(inputs)

        resolved = resolve_params(
            defaults=metadata.defaults,
            session_ctx=self.session_ctx,
            query_params=inputs,
            session_rules=self.session_rules,
        )
        return resolved.params

    async def _run_via_engine(
        self,
        metadata: SkillMetadata,
        inputs: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Any:
        """Map skill → registered kernel, passing dependency outputs."""
        operator = metadata.operator
        if not operator:
            raise RuntimeError(
                f"Skill {metadata.skill_id!r} has no operator mapping "
                f"and its executor_fn raised NotImplementedError"
            )

        dep_outputs = [state[d] for d in metadata.dependencies if d in state]

        resolved = self._resolve_skill_params(metadata, inputs)

        exec_node = ExecutionNode(
            node_id=metadata.skill_id,
            operator=operator,
            params=resolved,
            deps=[],  # deps already resolved by stage ordering
        )

        kernel = self.engine._kernels.get(operator)
        if kernel is None:
            raise RuntimeError(f"No kernel registered for operator {operator!r}")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, kernel, exec_node, dep_outputs)

    # ------------------------------------------------------------------
    # checkpoint / rollback
    # ------------------------------------------------------------------

    def rollback_to(self, stage_id: int) -> Optional[CheckpointState]:
        """Return the checkpoint for the given stage, or ``None``."""
        for cp in reversed(self.checkpoints):
            if cp.stage_id == stage_id:
                return cp
        return None
