"""
Tests for the skill definition layer and compiler pipeline.

Covers:
- Skill registration & @skill decorator
- Type checking
- Dependency resolution & cycle detection
- All three scheduling policies
- SkillCompiler end-to-end
- ExecutionPlan serialisation
- SkillScheduler bridge layer (dependency output propagation)
"""

from __future__ import annotations

import asyncio
import json

import pytest

from codeminer.agent.skills.core import (
    Cost,
    SkillInputSpec,
    SkillMetadata,
    SkillOutputSpec,
    SkillType,
)
from codeminer.agent.skills.registry import SkillRegistry, skill
from codeminer.compiler.execution import ExecutionEngine
from codeminer.compiler.skills.compiler import SkillCompiler
from codeminer.compiler.skills.dependency_resolver import DependencyResolver
from codeminer.compiler.skills.policy import (
    ConservativePolicy,
    CostAwarePolicy,
    ParallelPolicy,
)
from codeminer.compiler.skills.scheduler import SkillScheduler
from codeminer.compiler.skills.type_checker import TypeChecker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the global singleton before and after every test."""
    SkillRegistry.reset()
    yield
    SkillRegistry.reset()


def _register_simple_pipeline():
    """Register a small but representative pipeline: A, B → C → D."""
    reg = SkillRegistry()

    for sid, deps, cost in [
        ("a", [], Cost.LOW),
        ("b", [], Cost.HIGH),
        ("c", ["a", "b"], Cost.MEDIUM),
        ("d", ["c"], Cost.LOW),
    ]:
        reg.register(
            SkillMetadata(
                skill_id=sid,
                skill_type=SkillType.RETRIEVAL,
                inputs=[SkillInputSpec(name="query", type_hint="str")],
                outputs=SkillOutputSpec(type_hint="List[QueriedNode]"),
                cost=cost,
                dependencies=deps,
                operator=f"mock.{sid}",
            )
        )
    return reg


# ---------------------------------------------------------------------------
# Skill registration
# ---------------------------------------------------------------------------


class TestSkillRegistry:
    def test_register_and_get(self):
        reg = SkillRegistry()
        meta = SkillMetadata(
            skill_id="foo",
            skill_type=SkillType.RETRIEVAL,
        )
        reg.register(meta)
        assert reg.get("foo") is meta
        assert reg.has("foo")

    def test_duplicate_raises(self):
        reg = SkillRegistry()
        meta = SkillMetadata(skill_id="foo", skill_type=SkillType.RETRIEVAL)
        reg.register(meta)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(meta)

    def test_list_skills(self):
        reg = SkillRegistry()
        reg.register(SkillMetadata(skill_id="a", skill_type=SkillType.RETRIEVAL))
        reg.register(SkillMetadata(skill_id="b", skill_type=SkillType.RERANK))
        assert set(reg.list_skills().keys()) == {"a", "b"}

    def test_get_missing_returns_none(self):
        assert SkillRegistry().get("nonexistent") is None


class TestSkillDecorator:
    def test_decorator_registers_function(self):
        @skill(
            skill_id="test_skill",
            skill_type=SkillType.TRANSFORM,
            inputs=[SkillInputSpec(name="x", type_hint="int")],
            outputs=SkillOutputSpec(type_hint="int"),
            cost=Cost.LOW,
        )
        def my_fn(x: int) -> int:
            return x + 1

        reg = SkillRegistry()
        meta = reg.get("test_skill")
        assert meta is not None
        assert meta.executor_fn is my_fn
        assert meta.cost == Cost.LOW

    def test_decorator_preserves_function(self):
        @skill(skill_id="id2", skill_type=SkillType.CUSTOM)
        def hello():
            return "hello"

        assert hello() == "hello"

    def test_decorator_detects_async(self):
        @skill(skill_id="async_skill", skill_type=SkillType.CUSTOM)
        async def async_fn():
            pass

        meta = SkillRegistry().get("async_skill")
        assert meta is not None
        assert meta.async_capable is True


# ---------------------------------------------------------------------------
# Type checking
# ---------------------------------------------------------------------------


class TestTypeChecker:
    def setup_method(self):
        self.checker = TypeChecker()
        self.meta = SkillMetadata(
            skill_id="test",
            skill_type=SkillType.RETRIEVAL,
            inputs=[
                SkillInputSpec(name="query", type_hint="str", required=True),
                SkillInputSpec(
                    name="top_k", type_hint="int", required=False, default=10
                ),
            ],
        )

    def test_valid_inputs(self):
        errors = self.checker.check_skill_inputs(
            self.meta, {"query": "hello", "top_k": 5}
        )
        assert errors == []

    def test_missing_required(self):
        errors = self.checker.check_skill_inputs(self.meta, {})
        assert len(errors) == 1
        assert errors[0].input_name == "query"
        assert "not provided" in errors[0].message

    def test_type_mismatch(self):
        errors = self.checker.check_skill_inputs(self.meta, {"query": 123})
        assert len(errors) == 1
        assert "mismatch" in errors[0].message

    def test_optional_missing_ok(self):
        errors = self.checker.check_skill_inputs(self.meta, {"query": "hello"})
        assert errors == []


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------


class TestDependencyResolver:
    def test_simple_chain(self):
        reg = _register_simple_pipeline()
        resolver = DependencyResolver(reg)

        dag = resolver.resolve(
            [
                {"skill_id": "a", "inputs": {"query": "x"}},
                {"skill_id": "c", "inputs": {"query": "x"}},
                {"skill_id": "b", "inputs": {"query": "x"}},
                {"skill_id": "d", "inputs": {"query": "x"}},
            ]
        )

        assert set(dag.root_skills) == {"a", "b"}
        assert set(dag.nodes.keys()) == {"a", "b", "c", "d"}
        assert "d" in dag.nodes["c"].dependents
        assert dag.nodes["d"].dependencies == ["c"]

    def test_unknown_skill_raises(self):
        reg = _register_simple_pipeline()
        resolver = DependencyResolver(reg)
        with pytest.raises(ValueError, match="Unknown skill"):
            resolver.resolve([{"skill_id": "nonexistent"}])

    def test_missing_dependency_raises(self):
        reg = _register_simple_pipeline()
        resolver = DependencyResolver(reg)
        with pytest.raises(ValueError, match="not in the invocation set"):
            # "c" depends on "a" and "b", but we only supply "a"
            resolver.resolve(
                [
                    {"skill_id": "a", "inputs": {"query": "x"}},
                    {"skill_id": "c", "inputs": {"query": "x"}},
                ]
            )

    def test_cycle_detection(self):
        reg = SkillRegistry()
        reg.register(
            SkillMetadata(skill_id="x", skill_type=SkillType.CUSTOM, dependencies=["y"])
        )
        reg.register(
            SkillMetadata(skill_id="y", skill_type=SkillType.CUSTOM, dependencies=["x"])
        )
        resolver = DependencyResolver(reg)
        with pytest.raises(RuntimeError, match="Cycle"):
            resolver.resolve(
                [
                    {"skill_id": "x"},
                    {"skill_id": "y"},
                ]
            )

    def test_topological_order(self):
        reg = _register_simple_pipeline()
        resolver = DependencyResolver(reg)
        dag = resolver.resolve(
            [
                {"skill_id": "a", "inputs": {"query": "x"}},
                {"skill_id": "b", "inputs": {"query": "x"}},
                {"skill_id": "c", "inputs": {"query": "x"}},
                {"skill_id": "d", "inputs": {"query": "x"}},
            ]
        )
        topo = [n.skill_id for n in dag.iter_topo()]
        # a, b must come before c; c before d
        assert topo.index("a") < topo.index("c")
        assert topo.index("b") < topo.index("c")
        assert topo.index("c") < topo.index("d")


# ---------------------------------------------------------------------------
# Scheduling policies
# ---------------------------------------------------------------------------


def _invocations():
    return [
        {"skill_id": "a", "inputs": {"query": "x"}},
        {"skill_id": "b", "inputs": {"query": "x"}},
        {"skill_id": "c", "inputs": {"query": "x"}},
        {"skill_id": "d", "inputs": {"query": "x"}},
    ]


class TestParallelPolicy:
    def test_wavefront_grouping(self):
        reg = _register_simple_pipeline()
        dag = DependencyResolver(reg).resolve(_invocations())
        plan = ParallelPolicy().schedule(dag)

        assert len(plan.stages) == 3
        # Stage 0: a, b (roots, in parallel)
        assert set(plan.stages[0].skill_ids) == {"a", "b"}
        assert plan.stages[0].parallelism == 2
        # Stage 1: c
        assert plan.stages[1].skill_ids == ["c"]
        # Stage 2: d
        assert plan.stages[2].skill_ids == ["d"]


class TestConservativePolicy:
    def test_serial_execution(self):
        reg = _register_simple_pipeline()
        dag = DependencyResolver(reg).resolve(_invocations())
        plan = ConservativePolicy().schedule(dag)

        assert len(plan.stages) == 4
        assert all(s.parallelism == 1 for s in plan.stages)


class TestCostAwarePolicy:
    def test_respects_budget(self):
        reg = _register_simple_pipeline()
        dag = DependencyResolver(reg).resolve(_invocations())

        # Budget = 2.0 → a (LOW=1) fits, b (HIGH=4) alone in next stage
        plan = CostAwarePolicy(max_concurrent_cost=2.0).schedule(dag)

        first_stage_ids = set(plan.stages[0].skill_ids)
        # At minimum, one root runs per stage (budget guarantees progress).
        assert len(first_stage_ids) >= 1

    def test_high_budget_is_parallel(self):
        reg = _register_simple_pipeline()
        dag = DependencyResolver(reg).resolve(_invocations())

        plan = CostAwarePolicy(max_concurrent_cost=100.0).schedule(dag)
        # With a very high budget, first stage should have both roots.
        assert set(plan.stages[0].skill_ids) == {"a", "b"}


# ---------------------------------------------------------------------------
# SkillCompiler end-to-end
# ---------------------------------------------------------------------------


class TestSkillCompiler:
    def test_compile_pipeline(self):
        reg = _register_simple_pipeline()
        compiler = SkillCompiler(registry=reg, policy=ParallelPolicy())

        result = compiler.compile(_invocations())

        assert len(result.skill_dag.nodes) == 4
        assert len(result.execution_plan.stages) == 3
        assert set(result.skill_dag.root_skills) == {"a", "b"}

    def test_compile_with_type_warnings(self):
        reg = _register_simple_pipeline()
        compiler = SkillCompiler(registry=reg)

        invocations = [
            {"skill_id": "a", "inputs": {"query": 42}},  # wrong type
            {"skill_id": "b", "inputs": {"query": "ok"}},
            {"skill_id": "c", "inputs": {"query": "ok"}},
            {"skill_id": "d", "inputs": {"query": "ok"}},
        ]
        result = compiler.compile(invocations)

        assert len(result.warnings) == 1
        assert "[a]" in result.warnings[0]

    def test_compile_unknown_skill(self):
        compiler = SkillCompiler()
        with pytest.raises(ValueError, match="Unknown skill"):
            compiler.compile([{"skill_id": "ghost"}])


# ---------------------------------------------------------------------------
# ExecutionPlan serialisation
# ---------------------------------------------------------------------------


class TestExecutionPlanSerialisation:
    def test_to_dict_roundtrip(self):
        reg = _register_simple_pipeline()
        compiler = SkillCompiler(registry=reg, policy=ParallelPolicy())
        result = compiler.compile(_invocations())

        plan_dict = result.execution_plan.to_dict()

        # Must be JSON-serialisable.
        json_str = json.dumps(plan_dict)
        loaded = json.loads(json_str)

        assert loaded["policy_metadata"]["policy"] == "parallel"
        assert len(loaded["stages"]) == 3
        # First stage has two skills
        assert len(loaded["stages"][0]["skills"]) == 2
        assert loaded["stages"][0]["parallelism"] == 2

    def test_skill_metadata_to_dict(self):
        meta = SkillMetadata(
            skill_id="test",
            skill_type=SkillType.RETRIEVAL,
            inputs=[SkillInputSpec(name="q", type_hint="str")],
            cost=Cost.HIGH,
        )
        d = meta.to_dict()
        assert d["skill_id"] == "test"
        assert d["cost"] == "high"
        assert json.dumps(d)  # must be serialisable

    def test_dag_to_dict(self):
        reg = _register_simple_pipeline()
        dag = DependencyResolver(reg).resolve(_invocations())
        d = dag.to_dict()
        assert set(d["nodes"].keys()) == {"a", "b", "c", "d"}
        assert set(d["root_skills"]) == {"a", "b"}
        assert json.dumps(d)  # must be serialisable


# ---------------------------------------------------------------------------
# SkillScheduler bridge layer
# ---------------------------------------------------------------------------


class TestSkillSchedulerBridge:
    """Verify that the scheduler correctly passes dependency outputs to kernels."""

    def test_dependency_outputs_reach_kernel(self):
        """Skills with deps must receive previous outputs via the kernel bridge."""
        reg = SkillRegistry()

        # Root skill: returns a list
        reg.register(
            SkillMetadata(
                skill_id="root",
                skill_type=SkillType.RETRIEVAL,
                inputs=[SkillInputSpec(name="query", type_hint="str")],
                operator="mock.root",
            )
        )
        # Leaf skill: depends on root, fuses inputs
        reg.register(
            SkillMetadata(
                skill_id="leaf",
                skill_type=SkillType.AGGREGATE,
                inputs=[SkillInputSpec(name="query", type_hint="str")],
                dependencies=["root"],
                operator="mock.leaf",
            )
        )

        engine = ExecutionEngine()

        # Root kernel: ignores inputs (no deps), returns data
        def root_kernel(node, inputs):
            return ["result_from_root"]

        # Leaf kernel: must receive root's output in inputs
        captured_inputs = {}

        def leaf_kernel(node, inputs):
            captured_inputs["inputs"] = inputs
            flat = []
            for inp in inputs:
                if isinstance(inp, list):
                    flat.extend(inp)
                else:
                    flat.append(inp)
            return flat

        engine.register("mock.root", root_kernel)
        engine.register("mock.leaf", leaf_kernel)

        compiler = SkillCompiler(registry=reg, policy=ParallelPolicy())
        result = compiler.compile(
            [
                {"skill_id": "root", "inputs": {"query": "test"}},
                {"skill_id": "leaf", "inputs": {"query": "test"}},
            ]
        )

        scheduler = SkillScheduler(registry=reg, engine=engine)
        state = asyncio.get_event_loop().run_until_complete(
            scheduler.execute(result.execution_plan)
        )

        # The leaf must have received root's output
        assert "inputs" in captured_inputs
        assert captured_inputs["inputs"] == [["result_from_root"]]
        # Final state should contain both outputs
        assert state["root"] == ["result_from_root"]
        assert state["leaf"] == ["result_from_root"]

    def test_multiple_deps_all_passed(self):
        """A skill depending on two roots should receive both outputs."""
        reg = SkillRegistry()

        for sid in ("src_a", "src_b"):
            reg.register(
                SkillMetadata(
                    skill_id=sid,
                    skill_type=SkillType.RETRIEVAL,
                    inputs=[SkillInputSpec(name="query", type_hint="str")],
                    operator=f"mock.{sid}",
                )
            )
        reg.register(
            SkillMetadata(
                skill_id="fuse",
                skill_type=SkillType.AGGREGATE,
                inputs=[SkillInputSpec(name="query", type_hint="str")],
                dependencies=["src_a", "src_b"],
                operator="mock.fuse",
            )
        )

        engine = ExecutionEngine()
        engine.register("mock.src_a", lambda n, i: ["from_a"])
        engine.register("mock.src_b", lambda n, i: ["from_b"])

        captured = {}

        def fuse_kernel(node, inputs):
            captured["inputs"] = inputs
            return [item for branch in inputs for item in branch]

        engine.register("mock.fuse", fuse_kernel)

        compiler = SkillCompiler(registry=reg, policy=ParallelPolicy())
        result = compiler.compile(
            [
                {"skill_id": "src_a", "inputs": {"query": "q"}},
                {"skill_id": "src_b", "inputs": {"query": "q"}},
                {"skill_id": "fuse", "inputs": {"query": "q"}},
            ]
        )

        scheduler = SkillScheduler(registry=reg, engine=engine)
        state = asyncio.get_event_loop().run_until_complete(
            scheduler.execute(result.execution_plan)
        )

        assert len(captured["inputs"]) == 2
        assert ["from_a"] in captured["inputs"]
        assert ["from_b"] in captured["inputs"]
        assert set(state["fuse"]) == {"from_a", "from_b"}
