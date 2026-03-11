"""
Scheduling policies and the ``ExecutionPlan`` data structure.

``ExecutionPlan`` is the serialisation contract between the Python prototype
and the future Rust compiler — ``to_dict()`` **must** remain stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set

from ...agent.skills.core import COST_WEIGHTS
from .dependency_resolver import SkillDAG

# ---------------------------------------------------------------------------
# Execution plan data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExecutionStage:
    """A group of skills that can execute concurrently."""

    stage_id: int
    skill_ids: List[str] = field(default_factory=list)
    parallelism: int = 1


@dataclass(slots=True)
class ExecutionPlan:
    """
    Serialisable execution plan produced by the compiler.

    ``to_dict()`` is the Python↔Rust JSON contract.
    """

    stages: List[ExecutionStage] = field(default_factory=list)
    dag: Optional[SkillDAG] = field(default=None, repr=False)
    policy_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        dag_ref = self.dag
        return {
            "stages": [
                {
                    "stage_id": s.stage_id,
                    "skills": [
                        {
                            "skill_id": sid,
                            "inputs": dag_ref.nodes[sid].inputs if dag_ref else {},
                            "dependencies": (
                                dag_ref.nodes[sid].dependencies if dag_ref else []
                            ),
                        }
                        for sid in s.skill_ids
                    ],
                    "parallelism": s.parallelism,
                }
                for s in self.stages
            ],
            "policy_metadata": self.policy_metadata,
        }


# ---------------------------------------------------------------------------
# Policy protocol
# ---------------------------------------------------------------------------


class SchedulingPolicy(Protocol):
    """Interface that every scheduling strategy must satisfy."""

    def schedule(self, dag: SkillDAG) -> ExecutionPlan: ...


# ---------------------------------------------------------------------------
# Built-in policies
# ---------------------------------------------------------------------------


class ParallelPolicy:
    """Maximise concurrency — group skills by dependency wavefront."""

    def schedule(self, dag: SkillDAG) -> ExecutionPlan:
        plan = ExecutionPlan(dag=dag, policy_metadata={"policy": "parallel"})
        scheduled: Set[str] = set()
        stage_id = 0

        while len(scheduled) < len(dag.nodes):
            ready = [
                sid
                for sid, n in dag.nodes.items()
                if sid not in scheduled and all(d in scheduled for d in n.dependencies)
            ]
            if not ready:
                raise RuntimeError("DAG scheduling deadlock — possible cycle")

            plan.stages.append(
                ExecutionStage(
                    stage_id=stage_id,
                    skill_ids=ready,
                    parallelism=len(ready),
                )
            )
            scheduled.update(ready)
            stage_id += 1

        return plan


class ConservativePolicy:
    """Execute one skill at a time in topological order."""

    def schedule(self, dag: SkillDAG) -> ExecutionPlan:
        plan = ExecutionPlan(dag=dag, policy_metadata={"policy": "conservative"})
        for idx, node in enumerate(dag.iter_topo()):
            plan.stages.append(
                ExecutionStage(
                    stage_id=idx,
                    skill_ids=[node.skill_id],
                    parallelism=1,
                )
            )
        return plan


class CostAwarePolicy:
    """
    Respect a per-stage cost budget while still exploiting parallelism.

    Skills whose ``cost`` exceeds the remaining budget are deferred to the
    next stage.
    """

    def __init__(self, max_concurrent_cost: float = 5.0) -> None:
        self.max_concurrent_cost = max_concurrent_cost

    def schedule(self, dag: SkillDAG) -> ExecutionPlan:
        plan = ExecutionPlan(
            dag=dag,
            policy_metadata={
                "policy": "cost_aware",
                "max_concurrent_cost": self.max_concurrent_cost,
            },
        )

        scheduled: Set[str] = set()
        stage_id = 0

        while len(scheduled) < len(dag.nodes):
            ready = [
                sid
                for sid, n in dag.nodes.items()
                if sid not in scheduled and all(d in scheduled for d in n.dependencies)
            ]
            if not ready:
                raise RuntimeError("DAG scheduling deadlock — possible cycle")

            # Pack into stage up to budget.
            stage_skills: List[str] = []
            budget = 0.0
            for sid in ready:
                w = COST_WEIGHTS.get(dag.nodes[sid].metadata.cost, 2.0)
                if budget + w <= self.max_concurrent_cost or not stage_skills:
                    stage_skills.append(sid)
                    budget += w

            plan.stages.append(
                ExecutionStage(
                    stage_id=stage_id,
                    skill_ids=stage_skills,
                    parallelism=len(stage_skills),
                )
            )
            scheduled.update(stage_skills)
            stage_id += 1

        return plan
