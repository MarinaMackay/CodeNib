from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..compiler.execution import ExecutionEngine
from ..compiler.ir_exec import ExecutionNode
from ..compiler.ir_physical import PhysicalOperator
from ..graph.code_graph import CodeGraph
from ..graph.roi_subgraph import ROISubgraph
from ..log_utils import get_logger
from ..types import NodeInfo, QueriedNode

logger = get_logger(__name__)


@dataclass
class ExpandContext:
    """Shared resources for graph expansion operators."""

    code_graph: Optional[CodeGraph] = None
    default_top_k: int = 50
    default_hops: int = 2
    default_direction: str = "both"
    default_method: str = "bfs"
    default_damping: float = 0.85
    filter_tests: bool = True


def register_expand_ops(engine: ExecutionEngine, context: ExpandContext) -> None:
    """Register graph-expansion execution kernels on the provided engine."""
    engine.register(PhysicalOperator.GRAPH_WALK.value, _graph_walk_kernel(context))


def _graph_walk_kernel(context: ExpandContext):
    """Return a kernel that expands seed nodes along graph edges."""

    def run(node: ExecutionNode, inputs: List[object]) -> List[QueriedNode]:
        if context.code_graph is None:
            raise RuntimeError("graph.walk operator invoked but no CodeGraph provided.")

        # --- extract params ---
        params = node.params
        method = str(params.get("method", context.default_method))
        top_k = int(params.get("top_k", context.default_top_k))
        hops = int(params.get("hops", context.default_hops))
        direction = str(params.get("direction", context.default_direction))
        damping = float(params.get("damping", context.default_damping))
        filter_tests = bool(params.get("filter_tests", context.filter_tests))
        edge_types: Optional[List[str]] = params.get("edge_types")
        node_types: Optional[List[str]] = params.get("node_types")

        # --- extract seed node names from upstream inputs ---
        seed_names = _extract_seeds(inputs)
        if not seed_names:
            logger.warning("graph.walk: no seed nodes provided — returning empty")
            return []

        logger.debug(
            "graph.walk: method=%s seeds=%d top_k=%d hops=%d direction=%s",
            method,
            len(seed_names),
            top_k,
            hops,
            direction,
        )

        roi = ROISubgraph(context.code_graph)

        if method == "ppr":
            node_infos = roi.expand_ppr(
                node_names=seed_names,
                top_k=top_k,
                damping=damping,
                filter_tests=filter_tests,
            )
            # Exclude seeds from PPR results
            seed_set = set(seed_names)
            node_infos = [n for n in node_infos if n.node_name not in seed_set]
        else:
            # BFS k-hop expansion
            subgraph = roi.extract_subgraph(
                node_names=seed_names,
                k_hop=hops,
                edge_types=edge_types,
                direction=direction,
            )
            node_infos = roi.get_filtered_subgraph_nodes(
                subgraph,
                exclude_nodes=seed_names,
                filter_tests=filter_tests,
                node_types=node_types,
            )

        # Convert NodeInfo -> QueriedNode
        results = _nodeinfo_to_queried(node_infos)
        return results[:top_k]

    return run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_seeds(inputs: List[object]) -> List[str]:
    """Flatten upstream inputs into a list of seed node names."""
    names: List[str] = []
    for inp in inputs:
        if isinstance(inp, (list, tuple)):
            for item in inp:
                name = _get_node_name(item)
                if name:
                    names.append(name)
        else:
            name = _get_node_name(inp)
            if name:
                names.append(name)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique


def _get_node_name(item: object) -> Optional[str]:
    """Extract node_name from a QueriedNode/NodeInfo or plain string."""
    if isinstance(item, str):
        return item
    if hasattr(item, "node_name"):
        return item.node_name  # type: ignore[union-attr]
    return None


def _nodeinfo_to_queried(nodes: List[NodeInfo]) -> List[QueriedNode]:
    """Convert NodeInfo list to QueriedNode list, assigning rank-based scores."""
    results: List[QueriedNode] = []
    for rank, ni in enumerate(nodes):
        score = ni.score if ni.score is not None else 1.0 / (rank + 1)
        results.append(
            QueriedNode(
                node_name=ni.node_name,
                type=ni.type,
                file=ni.file,
                start_line=ni.start_line,
                end_line=ni.end_line,
                score=score,
                content=ni.content,
            )
        )
    return results
