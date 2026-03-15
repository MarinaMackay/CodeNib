"""Tests for graph expansion ops (codeminer.ops.expand)."""

from __future__ import annotations

import pytest

from codeminer.compiler.execution import ExecutionEngine
from codeminer.compiler.ir_exec import ExecutionNode
from codeminer.compiler.ir_physical import PhysicalOperator
from codeminer.graph.code_graph import CodeGraph
from codeminer.ops.expand import (
    ExpandContext,
    _extract_seeds,
    _nodeinfo_to_queried,
    register_expand_ops,
)
from codeminer.types import (
    EDGE_TYPE_REFERENCE,
    NODE_TYPE_CLASS,
    NODE_TYPE_FUNCTION,
    NODE_TYPE_METHOD,
    NodeInfo,
    QueriedNode,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_graph(tmp_path):
    """Build a small synthetic CodeGraph for testing.

    Structure:
        src/app.py (file)
          ├── AppClass (class, lines 0-50)
          │     ├── AppClass.run (method, lines 5-20)
          │     └── AppClass.stop (method, lines 25-45)
          └── helper_fn (function, lines 55-70)

        src/utils.py (file)
          └── parse_config (function, lines 0-30)

    Edges:
        AppClass.run --reference--> parse_config
        helper_fn --reference--> AppClass.stop
    """
    # Write dummy source files so get_node_content can read them
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    app_lines = [""] * 80
    app_lines[0] = "class AppClass:"
    app_lines[5] = "    def run(self):"
    app_lines[6] = "        pass"
    app_lines[25] = "    def stop(self):"
    app_lines[26] = "        pass"
    app_lines[55] = "def helper_fn():"
    app_lines[56] = "    pass"
    (src_dir / "app.py").write_text("\n".join(app_lines))

    utils_lines = [""] * 40
    utils_lines[0] = "def parse_config():"
    utils_lines[1] = "    pass"
    (src_dir / "utils.py").write_text("\n".join(utils_lines))

    g = CodeGraph(project_root=str(tmp_path))

    # --- src/app.py ---
    g.add_file_node("src/app.py")
    g.add_symbol_node(
        "src/app.py:AppClass",
        line=0,
        scope_start_line=0,
        scope_end_line=50,
        symbol_type=NODE_TYPE_CLASS,
    )
    g.add_containment_edge("src/app.py:AppClass")

    g.update_current_scope("src/app.py:AppClass", 0, 50)
    g.add_symbol_node(
        "src/app.py:AppClass.run()",
        line=5,
        scope_start_line=5,
        scope_end_line=20,
        symbol_type=NODE_TYPE_METHOD,
    )
    g.add_containment_edge("src/app.py:AppClass.run()")

    g.add_symbol_node(
        "src/app.py:AppClass.stop()",
        line=25,
        scope_start_line=25,
        scope_end_line=45,
        symbol_type=NODE_TYPE_METHOD,
    )
    g.add_containment_edge("src/app.py:AppClass.stop()")

    # Reset scope to file for helper_fn
    g.scope_stack = [{"src/app.py": None}]
    g.current_scope = "src/app.py"
    g.add_symbol_node(
        "src/app.py:helper_fn()",
        line=55,
        scope_start_line=55,
        scope_end_line=70,
        symbol_type=NODE_TYPE_FUNCTION,
    )
    g.add_containment_edge("src/app.py:helper_fn()")

    # --- src/utils.py ---
    g.add_file_node("src/utils.py")
    g.add_symbol_node(
        "src/utils.py:parse_config()",
        line=0,
        scope_start_line=0,
        scope_end_line=30,
        symbol_type=NODE_TYPE_FUNCTION,
    )
    g.add_containment_edge("src/utils.py:parse_config()")

    # --- Reference edges ---
    g.current_scope = "src/app.py:AppClass.run()"
    g.add_symbol_reference("src/utils.py:parse_config()")

    g.current_scope = "src/app.py:helper_fn()"
    g.add_symbol_reference("src/app.py:AppClass.stop()")

    return g


@pytest.fixture()
def expand_context(sample_graph):
    return ExpandContext(code_graph=sample_graph, default_top_k=50, default_hops=2)


@pytest.fixture()
def engine_with_expand(expand_context):
    engine = ExecutionEngine()
    register_expand_ops(engine, expand_context)
    return engine


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestExtractSeeds:
    def test_from_queried_nodes(self):
        nodes = [
            QueriedNode(node_name="a"),
            QueriedNode(node_name="b"),
        ]
        assert _extract_seeds([nodes]) == ["a", "b"]

    def test_from_strings(self):
        assert _extract_seeds([["x", "y"]]) == ["x", "y"]

    def test_deduplicates(self):
        nodes = [QueriedNode(node_name="a"), QueriedNode(node_name="a")]
        assert _extract_seeds([nodes]) == ["a"]

    def test_empty_input(self):
        assert _extract_seeds([]) == []
        assert _extract_seeds([[]]) == []


class TestNodeInfoToQueried:
    def test_preserves_ppr_score(self):
        infos = [NodeInfo(node_name="a", type="function", score=0.42)]
        result = _nodeinfo_to_queried(infos)
        assert len(result) == 1
        assert isinstance(result[0], QueriedNode)
        assert result[0].score == 0.42

    def test_assigns_rank_score_when_missing(self):
        infos = [
            NodeInfo(node_name="a", type="function"),
            NodeInfo(node_name="b", type="function"),
        ]
        result = _nodeinfo_to_queried(infos)
        assert result[0].score == 1.0  # 1/(0+1)
        assert result[1].score == 0.5  # 1/(1+1)


# ---------------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------------


class TestGraphWalkKernel:
    def _make_exec_node(self, **params):
        return ExecutionNode(
            node_id="test_expand",
            operator=PhysicalOperator.GRAPH_WALK.value,
            params=params,
            deps=[],
        )

    def test_bfs_1hop_from_class(self, engine_with_expand):
        """BFS 1-hop from AppClass should find its methods and the file."""
        node = self._make_exec_node(method="bfs", hops=1, direction="both")
        seeds = [QueriedNode(node_name="src/app.py:AppClass")]
        kernel = engine_with_expand._kernels[PhysicalOperator.GRAPH_WALK.value]
        results = kernel(node, [seeds])

        result_names = {r.node_name for r in results}
        # AppClass is the seed, should be excluded
        assert "src/app.py:AppClass" not in result_names
        # Methods are 1-hop via containment
        assert "src/app.py:AppClass.run()" in result_names
        assert "src/app.py:AppClass.stop()" in result_names

    def test_bfs_2hop_reaches_cross_file_reference(self, engine_with_expand):
        """BFS 2-hop from AppClass should reach parse_config via run()."""
        node = self._make_exec_node(method="bfs", hops=2, direction="both")
        seeds = [QueriedNode(node_name="src/app.py:AppClass")]
        kernel = engine_with_expand._kernels[PhysicalOperator.GRAPH_WALK.value]
        results = kernel(node, [seeds])

        result_names = {r.node_name for r in results}
        assert "src/utils.py:parse_config()" in result_names

    def test_bfs_forward_only(self, engine_with_expand):
        """Forward-only BFS from run() should reach parse_config but not AppClass."""
        node = self._make_exec_node(method="bfs", hops=1, direction="forward")
        seeds = [QueriedNode(node_name="src/app.py:AppClass.run()")]
        kernel = engine_with_expand._kernels[PhysicalOperator.GRAPH_WALK.value]
        results = kernel(node, [seeds])

        result_names = {r.node_name for r in results}
        assert "src/utils.py:parse_config()" in result_names

    def test_bfs_edge_type_filter(self, engine_with_expand):
        """BFS with reference-only edges from AppClass should
        not find methods (contain edges)."""
        node = self._make_exec_node(
            method="bfs",
            hops=2,
            direction="both",
            edge_types=[EDGE_TYPE_REFERENCE],
        )
        seeds = [QueriedNode(node_name="src/app.py:AppClass")]
        kernel = engine_with_expand._kernels[PhysicalOperator.GRAPH_WALK.value]
        results = kernel(node, [seeds])

        result_names = {r.node_name for r in results}
        # Methods are connected via CONTAIN, not REFERENCE
        assert "src/app.py:AppClass.run()" not in result_names
        assert "src/app.py:AppClass.stop()" not in result_names

    def test_ppr_returns_scored_results(self, engine_with_expand):
        """PPR expansion should return scored results."""
        node = self._make_exec_node(method="ppr", top_k=10)
        seeds = [QueriedNode(node_name="src/app.py:AppClass.run()")]
        kernel = engine_with_expand._kernels[PhysicalOperator.GRAPH_WALK.value]
        results = kernel(node, [seeds])

        assert len(results) > 0
        assert all(isinstance(r, QueriedNode) for r in results)
        assert all(r.score is not None and r.score >= 0 for r in results)
        # Seed should be excluded
        assert all(r.node_name != "src/app.py:AppClass.run()" for r in results)

    def test_top_k_limits_results(self, engine_with_expand):
        """top_k should limit the number of returned results."""
        node = self._make_exec_node(method="bfs", hops=3, top_k=2)
        seeds = [QueriedNode(node_name="src/app.py:AppClass")]
        kernel = engine_with_expand._kernels[PhysicalOperator.GRAPH_WALK.value]
        results = kernel(node, [seeds])

        assert len(results) <= 2

    def test_no_graph_raises(self):
        """Kernel should raise RuntimeError when no graph is provided."""
        ctx = ExpandContext(code_graph=None)
        engine = ExecutionEngine()
        register_expand_ops(engine, ctx)

        node = ExecutionNode(
            node_id="test",
            operator=PhysicalOperator.GRAPH_WALK.value,
            params={"method": "bfs"},
            deps=[],
        )
        kernel = engine._kernels[PhysicalOperator.GRAPH_WALK.value]
        with pytest.raises(RuntimeError, match="no CodeGraph"):
            kernel(node, [[QueriedNode(node_name="x")]])

    def test_empty_seeds_returns_empty(self, engine_with_expand):
        """Empty seed list should return empty results."""
        node = self._make_exec_node(method="bfs", hops=1)
        kernel = engine_with_expand._kernels[PhysicalOperator.GRAPH_WALK.value]
        results = kernel(node, [[]])
        assert results == []

    def test_results_include_content(self, engine_with_expand):
        """Expanded nodes should include source content."""
        node = self._make_exec_node(method="bfs", hops=1, direction="both")
        seeds = [QueriedNode(node_name="src/app.py:AppClass")]
        kernel = engine_with_expand._kernels[PhysicalOperator.GRAPH_WALK.value]
        results = kernel(node, [seeds])

        # At least some results should have content
        results_with_content = [r for r in results if r.content]
        assert len(results_with_content) > 0


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------


class TestRegisterExpandOps:
    def test_registers_graph_walk(self):
        ctx = ExpandContext(code_graph=None)
        engine = ExecutionEngine()
        register_expand_ops(engine, ctx)
        assert PhysicalOperator.GRAPH_WALK.value in engine._kernels
