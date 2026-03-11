"""
Skill compilation components.

This sub-package contains the compiler pipeline that turns declarative skill
invocations into a serialisable ``ExecutionPlan``.  In Phase 2 the internals
will be replaced by a Rust binary communicating via JSON.
"""

from .compiler import CompilationResult, SkillCompiler
from .dependency_resolver import DependencyResolver, SkillDAG, SkillNode
from .policy import (
    ConservativePolicy,
    CostAwarePolicy,
    ExecutionPlan,
    ExecutionStage,
    ParallelPolicy,
    SchedulingPolicy,
)
from .type_checker import TypeChecker, TypeCheckError

__all__ = [
    "CompilationResult",
    "ConservativePolicy",
    "CostAwarePolicy",
    "DependencyResolver",
    "ExecutionPlan",
    "ExecutionStage",
    "ParallelPolicy",
    "SchedulingPolicy",
    "SkillCompiler",
    "SkillDAG",
    "SkillNode",
    "TypeCheckError",
    "TypeChecker",
]
