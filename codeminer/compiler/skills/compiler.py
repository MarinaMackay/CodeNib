"""
Skill compiler — the main orchestrator.

Runs the compilation pipeline:
    type check → resource resolution → dependency resolution → policy scheduling → ExecutionPlan
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ...agent.skills.registry import SkillRegistry
from ..resources import IndexRequirement, ResourcePlan, ResourceResolver
from .dependency_resolver import DependencyResolver, SkillDAG
from .policy import ExecutionPlan, ParallelPolicy, SchedulingPolicy
from .type_checker import TypeChecker


@dataclass(slots=True)
class CompilationResult:
    """Output of a successful compilation pass."""

    execution_plan: ExecutionPlan
    skill_dag: SkillDAG
    warnings: List[str] = field(default_factory=list)
    resource_plan: Optional[ResourcePlan] = None


class SkillCompiler:
    """
    Compile skill invocations into a serialisable ``ExecutionPlan``.

    In the Rust version this becomes the core binary; the Python scheduler
    consumes the plan it emits.
    """

    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        type_checker: Optional[TypeChecker] = None,
        dependency_resolver: Optional[DependencyResolver] = None,
        policy: Optional[SchedulingPolicy] = None,
        *,
        resource_resolver: Optional[ResourceResolver] = None,
    ) -> None:
        self.registry = registry or SkillRegistry()
        self.type_checker = type_checker or TypeChecker()
        self.dependency_resolver = dependency_resolver or DependencyResolver(
            self.registry
        )
        self.policy: SchedulingPolicy = policy or ParallelPolicy()
        self.resource_resolver = resource_resolver

    def compile(
        self,
        skill_invocations: List[Dict[str, Any]],
    ) -> CompilationResult:
        """
        Run the full compilation pipeline.

        Args:
            skill_invocations: Each dict must contain ``"skill_id"`` and may
                contain ``"inputs"`` and ``"depends_on"``.

        Returns:
            A ``CompilationResult`` holding the execution plan, the resolved
            DAG, and any type-checking warnings.
        """
        warnings: List[str] = []

        # Phase 1 — type checking
        for inv in skill_invocations:
            skill_id = inv["skill_id"]
            metadata = self.registry.get(skill_id)
            if metadata is None:
                raise ValueError(f"Unknown skill: {skill_id!r}")
            errors = self.type_checker.check_skill_inputs(
                metadata, inv.get("inputs", {})
            )
            for err in errors:
                warnings.append(f"[{skill_id}] {err.message}")

        # Phase 2 — dependency resolution
        skill_dag = self.dependency_resolver.resolve(skill_invocations)

        # Phase 2.5 — resource resolution (index freshness checks)
        resource_plan: Optional[ResourcePlan] = None
        if self.resource_resolver is not None:
            requirements = self._collect_index_requirements(skill_dag)
            if requirements:
                resource_plan = self.resource_resolver.resolve(requirements)
                warnings.extend(resource_plan.warnings)

        # Phase 3 — scheduling policy
        execution_plan = self.policy.schedule(skill_dag)

        return CompilationResult(
            execution_plan=execution_plan,
            skill_dag=skill_dag,
            warnings=warnings,
            resource_plan=resource_plan,
        )

    @staticmethod
    def _collect_index_requirements(
        dag: SkillDAG,
    ) -> List[IndexRequirement]:
        """Gather unique index requirements from all skills in the DAG."""
        seen: Set[Tuple[str, str]] = set()
        requirements: List[IndexRequirement] = []

        for node in dag.nodes.values():
            for req in getattr(node.metadata, "index_requirements", []):
                key = (req.index_type, req.scope)
                if key not in seen:
                    seen.add(key)
                    requirements.append(req)

        return requirements
