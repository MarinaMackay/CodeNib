"""
Dependency resolution and DAG construction for skills.

Builds a ``SkillDAG`` from a list of skill invocations, validating that all
declared dependencies exist and that no cycles are present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...agent.skills.core import SkillMetadata
from ...agent.skills.registry import SkillRegistry


@dataclass(slots=True)
class SkillNode:
    """A node in the skill dependency DAG."""

    skill_id: str
    metadata: SkillMetadata
    inputs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)


@dataclass(slots=True)
class SkillDAG:
    """
    Directed acyclic graph of skill nodes.

    Stored as adjacency lists — trivially serializable for the Rust handoff.
    """

    nodes: Dict[str, SkillNode] = field(default_factory=dict)
    root_skills: List[str] = field(default_factory=list)

    def iter_topo(self) -> List[SkillNode]:
        """Topological sort with cycle detection (Kahn-style DFS)."""
        visited: Dict[str, str] = {}
        order: List[str] = []

        def _dfs(skill_id: str) -> None:
            state = visited.get(skill_id)
            if state == "visiting":
                raise RuntimeError(f"Cycle detected in skill DAG at {skill_id!r}")
            if state == "done":
                return
            visited[skill_id] = "visiting"
            node = self.nodes[skill_id]
            for dep_id in node.dependencies:
                _dfs(dep_id)
            visited[skill_id] = "done"
            order.append(skill_id)

        for sid in self.nodes:
            _dfs(sid)

        return [self.nodes[sid] for sid in order]

    # ---- serialisation ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {
                sid: {
                    "skill_id": n.skill_id,
                    "inputs": n.inputs,
                    "dependencies": n.dependencies,
                    "dependents": n.dependents,
                }
                for sid, n in self.nodes.items()
            },
            "root_skills": list(self.root_skills),
        }


class DependencyResolver:
    """
    Analyse skill invocations and build a ``SkillDAG``.

    In Rust: graph algorithms with proper cycle detection and error spans.
    """

    def __init__(self, registry: Optional[SkillRegistry] = None) -> None:
        self.registry = registry or SkillRegistry()

    def resolve(self, skill_invocations: List[Dict[str, Any]]) -> SkillDAG:
        """
        Build a DAG from skill invocation dicts.

        Each invocation must have at least ``skill_id``.  Optional keys:
        ``inputs`` (dict), ``depends_on`` (list of skill_id strings).
        """
        dag = SkillDAG()

        # Phase 1 — create nodes
        for inv in skill_invocations:
            skill_id = inv["skill_id"]
            metadata = self.registry.get(skill_id)
            if metadata is None:
                raise ValueError(f"Unknown skill: {skill_id!r}")

            explicit_deps = inv.get("depends_on", [])
            # Merge explicit invocation deps with metadata-declared deps,
            # de-duplicated and order-preserved.
            all_deps = list(dict.fromkeys(explicit_deps + metadata.dependencies))

            dag.nodes[skill_id] = SkillNode(
                skill_id=skill_id,
                metadata=metadata,
                inputs=inv.get("inputs", {}),
                dependencies=all_deps,
            )

        # Phase 2 — wire dependents
        for skill_id, node in dag.nodes.items():
            for dep_id in node.dependencies:
                if dep_id not in dag.nodes:
                    raise ValueError(
                        f"Skill {skill_id!r} depends on {dep_id!r} "
                        f"which is not in the invocation set"
                    )
                dag.nodes[dep_id].dependents.append(skill_id)

        # Phase 3 — identify roots (no incoming edges)
        dag.root_skills = [sid for sid, n in dag.nodes.items() if not n.dependencies]

        # Validate no cycles (iter_topo raises on cycle).
        dag.iter_topo()

        return dag
