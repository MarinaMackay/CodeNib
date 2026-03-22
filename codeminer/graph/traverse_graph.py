# Revised from:
# https://github.com/All-Hands-AI/openhands-aci/blob/main/openhands_aci/indexing/locagent/repo/dependency_graph/traverse_graph.py

from typing import List, Optional

from ..utils import is_test_file
from .code_graph import CodeGraph


class RepoDependencySearcher:
    """Traverse Repository Graph using igraph"""

    def __init__(self, code_graph: CodeGraph):
        self.code_graph = code_graph
        self.graph = code_graph.get_graph()

    def subgraph(self, nids):
        """Create a subgraph from a list of node IDs"""
        vertex_ids = [
            self.code_graph.name_to_vertex[nid]
            for nid in nids
            if nid in self.code_graph.name_to_vertex
        ]
        return self.graph.subgraph(vertex_ids)

    def get_neighbors(
        self,
        nid,
        direction="forward",
        ntype_filter=None,
        etype_filter=None,
        ignore_test_file=True,
    ):
        """Get neighbors of a node with filtering options"""
        nodes, edges = [], []

        if nid not in self.code_graph.name_to_vertex:
            return nodes, edges

        vertex_id = self.code_graph.name_to_vertex[nid]

        if direction == "forward":
            neighbor_ids = self.graph.successors(vertex_id)
        elif direction == "backward":
            neighbor_ids = self.graph.predecessors(vertex_id)
        else:
            neighbor_ids = self.graph.neighbors(vertex_id)

        for neighbor_id in neighbor_ids:
            neighbor_vertex = self.graph.vs[neighbor_id]
            neighbor_nid = neighbor_vertex["name"]

            # Apply filters
            if ntype_filter and (
                "type" not in neighbor_vertex.attributes()
                or neighbor_vertex["type"] not in ntype_filter
            ):
                continue
            if ignore_test_file and is_test_file(neighbor_nid):
                continue

            # Get edge information
            if direction == "forward":
                edge_id = self.graph.get_eid(vertex_id, neighbor_id, error=False)
                if edge_id is not None:
                    edge = self.graph.es[edge_id]
                    etype = edge["type"] if "type" in edge.attributes() else "unknown"
                    if etype_filter and etype not in etype_filter:
                        continue
                    edges.append((nid, neighbor_nid, 0, {"type": etype}))
                    nodes.append(neighbor_nid)
            elif direction == "backward":
                edge_id = self.graph.get_eid(neighbor_id, vertex_id, error=False)
                if edge_id is not None:
                    edge = self.graph.es[edge_id]
                    etype = edge["type"] if "type" in edge.attributes() else "unknown"
                    if etype_filter and etype not in etype_filter:
                        continue
                    edges.append((neighbor_nid, nid, 0, {"type": etype}))
                    nodes.append(neighbor_nid)

        return nodes, edges


def traverse_tree_structure(
    code_graph: CodeGraph,
    root,
    direction="downstream",
    hops=2,
    node_type_filter: Optional[List[str]] = None,
    edge_type_filter: Optional[List[str]] = None,
):
    """
    Traverse tree structure starting from a root node

    Args:
        code_graph: CodeGraph instance
        root: Root node ID to start traversal from
        direction: 'downstream', 'upstream', or 'both'
        hops: Maximum number of hops to traverse (-1 for unlimited)
        node_type_filter: Filter by node types
        edge_type_filter: Filter by edge types

    Returns:
        String representation of the tree structure
    """
    if hops == -1:
        hops = 20

    if root not in code_graph.name_to_vertex:
        return f"Node {root!r} not found in graph"

    rtn_str = []
    traversed_nodes = set()
    traversed_edges = set()

    def traverse(node, prefix, is_last, level, edge_type, edirection):
        if level > hops:
            return

        if node == root and level == 0:
            rtn_str.append(f"{node}")
            new_prefix = ""
            edirection = direction
        else:
            connector = "└── " if is_last else "├── "
            connector += f"{edge_type} ── "
            rtn_str.append(f"{prefix}{connector}{node}")
            new_prefix = prefix + (" " if is_last else "│") + " " * (len(connector) - 1)

        if node in traversed_nodes:
            return
        traversed_nodes.add(node)

        # Separate containment and reference edges
        contain_neighbors = []  # (neighbor_id, etype, edir)
        reference_neighbors = []  # (neighbor_id, etype, edir)

        def is_ntype_not_valid(_ntype):
            return node_type_filter is not None and _ntype not in node_type_filter

        def is_etype_not_valid(_etype):
            return edge_type_filter is not None and _etype not in edge_type_filter

        if node not in code_graph.name_to_vertex:
            return

        vertex_id = code_graph.name_to_vertex[node]

        # Downstream traversal
        if "downstream" == edirection or (node == root and direction == "both"):
            neighbor_ids = code_graph.graph.successors(vertex_id)
            for neighbor_id in neighbor_ids:
                neighbor_vertex = code_graph.graph.vs[neighbor_id]
                neighbor = neighbor_vertex["name"]
                neigh_type = (
                    neighbor_vertex["type"]
                    if "type" in neighbor_vertex.attributes()
                    else "unknown"
                )

                if is_ntype_not_valid(neigh_type):
                    continue

                # Get edge type
                edge_id = code_graph.graph.get_eid(vertex_id, neighbor_id, error=False)
                if edge_id is not None:
                    etype = (
                        code_graph.graph.es[edge_id]["type"]
                        if "type" in code_graph.graph.es[edge_id].attributes()
                        else "unknown"
                    )
                    if is_etype_not_valid(etype):
                        continue
                    if not is_test_file(neighbor):
                        if (node, etype, neighbor) not in traversed_edges:
                            traversed_edges.add((node, etype, neighbor))
                            # Separate by edge type: "contain" vs others
                            if etype == "contain":
                                contain_neighbors.append(
                                    (neighbor, etype, "downstream")
                                )
                            else:
                                reference_neighbors.append(
                                    (neighbor, etype, "downstream")
                                )

        # Upstream traversal
        if "upstream" == edirection or (node == root and direction == "both"):
            neighbor_ids = code_graph.graph.predecessors(vertex_id)
            for neighbor_id in neighbor_ids:
                neighbor_vertex = code_graph.graph.vs[neighbor_id]
                neighbor = neighbor_vertex["name"]
                neigh_type = (
                    neighbor_vertex["type"]
                    if "type" in neighbor_vertex.attributes()
                    else "unknown"
                )

                if is_ntype_not_valid(neigh_type):
                    continue

                # Get edge type
                edge_id = code_graph.graph.get_eid(neighbor_id, vertex_id, error=False)
                if edge_id is not None:
                    etype = (
                        code_graph.graph.es[edge_id]["type"]
                        if "type" in code_graph.graph.es[edge_id].attributes()
                        else "unknown"
                    )
                    if is_etype_not_valid(etype):
                        continue
                    if not is_test_file(neighbor):
                        if (neighbor, etype, node) not in traversed_edges:
                            traversed_edges.add((neighbor, etype, node))
                            # Separate by edge type: "contain" vs others
                            if etype == "contain":
                                contain_neighbors.append((neighbor, etype, "upstream"))
                            else:
                                reference_neighbors.append(
                                    (neighbor, etype, "upstream")
                                )

        # Combine: containment first, then references
        all_neighbors = contain_neighbors + reference_neighbors

        for i, (neigh_id, etype, edir) in enumerate(all_neighbors):
            is_last_child = i == len(all_neighbors) - 1
            if edir == "upstream":
                etype += "-by"
            traverse(neigh_id, new_prefix, is_last_child, level + 1, etype, edir)

    traverse(root, "", False, 0, None, None)
    return "\n".join(rtn_str)
