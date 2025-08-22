import re
from collections import defaultdict
from typing import List, Optional

from .code_graph import CodeGraph
from .types import NODE_TYPE_DIRECTORY, NODE_TYPE_FILE, NODE_TYPE_SYMBOL


def is_test_file(nid):
    """Check if a node ID belongs to a test file"""
    if ":" in nid:
        file_path = nid.split(":")[0]
    else:
        file_path = nid
    word_list = re.split(r" |_|\/", file_path.lower())  # split by ' ', '_', and '/'
    return any([word.startswith("test") for word in word_list])


def wrap_code_snippet(code_snippet, start_line, end_line):
    """Wrap code snippet with line numbers"""
    lines = code_snippet.split("\n")
    max_line_number = start_line + len(lines) - 1
    if not start_line == end_line == 1:  # which is a file
        assert max_line_number == end_line
    number_width = len(str(max_line_number))
    return (
        "```\n"
        + "\n".join(
            f"{str(i + start_line).rjust(number_width)} | {line}"
            for i, line in enumerate(lines)
        )
        + "\n```"
    )


class RepoEntitySearcher:
    """Retrieve Entity IDs and Code Snippets from Repository Graph"""

    def __init__(self, code_graph: CodeGraph):
        self.code_graph = code_graph
        self.graph = code_graph.get_graph()
        self._global_name_dict = None
        self._global_name_dict_lowercase = None

    @property
    def global_name_dict(self):
        """Build a dictionary mapping names to node IDs"""
        if self._global_name_dict is None:
            _global_name_dict = defaultdict(list)
            for vertex in self.graph.vs:
                nid = vertex["name"]
                if is_test_file(nid):
                    continue

                if nid.endswith(".py"):
                    fname = nid.split("/")[-1]
                    _global_name_dict[fname].append(nid)

                    name = nid[: -(len(".py"))].split("/")[-1]
                    _global_name_dict[name].append(nid)

                elif ":" in nid:
                    name = nid.split(":")[-1].split(".")[-1]
                    _global_name_dict[name].append(nid)

            self._global_name_dict = _global_name_dict
        return self._global_name_dict

    @property
    def global_name_dict_lowercase(self):
        """Build a lowercase dictionary mapping names to node IDs"""
        if self._global_name_dict_lowercase is None:
            _global_name_dict_lowercase = defaultdict(list)
            for vertex in self.graph.vs:
                nid = vertex["name"]
                if is_test_file(nid):
                    continue

                if nid.endswith(".py"):
                    fname = nid.split("/")[-1].lower()
                    _global_name_dict_lowercase[fname].append(nid)

                    name = nid[: -(len(".py"))].split("/")[-1].lower()
                    _global_name_dict_lowercase[name].append(nid)

                elif ":" in nid:
                    name = nid.split(":")[-1].split(".")[-1].lower()
                    _global_name_dict_lowercase[name].append(nid)

            self._global_name_dict_lowercase = _global_name_dict_lowercase
        return self._global_name_dict_lowercase

    def has_node(self, nid, include_test=False):
        """Check if a node exists in the graph"""
        if not include_test and is_test_file(nid):
            return False
        return nid in self.code_graph.name_to_vertex

    def get_node_data(self, nids, return_code_content=False, wrap_with_ln=True):
        """Get formatted node data for a list of node IDs"""
        rtn = []
        for nid in nids:
            if nid not in self.code_graph.name_to_vertex:
                continue

            vertex_id = self.code_graph.name_to_vertex[nid]
            vertex = self.graph.vs[vertex_id]

            formatted_data = {
                "node_id": nid,
                "type": vertex.get("type", "unknown"),
            }

            # Get code content from the CodeGraph
            code_content = self.code_graph.get_node_content(vertex_id)
            if code_content:
                start_line = vertex.get("start_line", 1)
                end_line = vertex.get("end_line", 1)

                formatted_data["start_line"] = start_line
                formatted_data["end_line"] = end_line

                if vertex["type"] == NODE_TYPE_FILE:
                    end_line = len(code_content.split("\n"))
                    formatted_data["end_line"] = end_line

                if return_code_content and wrap_with_ln:
                    formatted_data["code_content"] = wrap_code_snippet(
                        code_content, start_line, end_line
                    )
                elif return_code_content:
                    formatted_data["code_content"] = code_content

            rtn.append(formatted_data)
        return rtn

    def get_all_nodes_by_type(self, node_type):
        """Get all nodes of a specific type"""
        valid_types = [NODE_TYPE_DIRECTORY, NODE_TYPE_FILE, NODE_TYPE_SYMBOL]
        assert node_type in valid_types

        nodes = []
        for vertex in self.graph.vs:
            nid = vertex["name"]
            if is_test_file(nid):
                continue
            if vertex.get("type") == node_type:
                if node_type == NODE_TYPE_FILE:
                    code_content = self.code_graph.get_node_content(vertex.index)
                    formatted_data = {
                        "name": nid,
                        "type": vertex["type"],
                        "content": code_content.split("\n") if code_content else [],
                    }
                elif node_type == NODE_TYPE_SYMBOL:
                    code_content = self.code_graph.get_node_content(vertex.index)
                    formatted_data = {
                        "name": nid.split(":")[-1] if ":" in nid else nid,
                        "file": vertex.get("file"),
                        "type": vertex["type"],
                        "content": code_content.split("\n") if code_content else [],
                        "start_line": vertex.get("start_line", 0),
                        "end_line": vertex.get("end_line", 0),
                    }
                elif node_type == NODE_TYPE_DIRECTORY:
                    formatted_data = {
                        "name": nid,
                        "type": vertex["type"],
                    }
                nodes.append(formatted_data)
        return nodes


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
            if ntype_filter and neighbor_vertex.get("type") not in ntype_filter:
                continue
            if ignore_test_file and is_test_file(neighbor_nid):
                continue

            # Get edge information
            if direction == "forward":
                edge_id = self.graph.get_eid(vertex_id, neighbor_id, error=False)
                if edge_id is not None:
                    edge = self.graph.es[edge_id]
                    etype = edge.get("type", "unknown")
                    if etype_filter and etype not in etype_filter:
                        continue
                    edges.append((nid, neighbor_nid, 0, {"type": etype}))
                    nodes.append(neighbor_nid)
            elif direction == "backward":
                edge_id = self.graph.get_eid(neighbor_id, vertex_id, error=False)
                if edge_id is not None:
                    edge = self.graph.es[edge_id]
                    etype = edge.get("type", "unknown")
                    if etype_filter and etype not in etype_filter:
                        continue
                    edges.append((neighbor_nid, nid, 0, {"type": etype}))
                    nodes.append(neighbor_nid)

        return nodes, edges


def traverse_graph_structure(
    code_graph: CodeGraph,
    roots,
    direction="downstream",
    hops=2,
    node_type_filter: Optional[List[str]] = None,
    edge_type_filter: Optional[List[str]] = None,
):
    """
    Traverse graph structure starting from root nodes

    Args:
        code_graph: CodeGraph instance
        roots: List of root node IDs to start traversal from
        direction: 'downstream', 'upstream', or 'both'
        hops: Maximum number of hops to traverse (-1 for unlimited)
        node_type_filter: Filter by node types
        edge_type_filter: Filter by edge types

    Returns:
        String representation of the graph structure
    """
    if hops == -1:
        hops = 20

    searcher = RepoDependencySearcher(code_graph)

    # Create a simple representation using lists instead of networkx
    visited_nodes = set()
    edges_list = []

    # BFS traversal
    frontiers = [(nid, 0) for nid in roots if nid in code_graph.name_to_vertex]

    while frontiers:
        nid, level = frontiers.pop(0)
        if nid in visited_nodes or abs(level) >= hops:
            continue
        visited_nodes.add(nid)

        ntype_filter, etype_filter = node_type_filter, edge_type_filter

        # Forward search
        if level > 0 or (
            level == 0 and (direction == "downstream" or direction == "both")
        ):
            nodes, edges = searcher.get_neighbors(
                nid,
                "forward",
                ntype_filter=ntype_filter,
                etype_filter=etype_filter,
                ignore_test_file=True,
            )
            frontiers.extend([(node, level + 1) for node in nodes])
            edges_list.extend(edges)

        # Backward search
        if level < 0 or (
            level == 0 and (direction == "upstream" or direction == "both")
        ):
            nodes, edges = searcher.get_neighbors(
                nid,
                "backward",
                ntype_filter=ntype_filter,
                etype_filter=etype_filter,
                ignore_test_file=True,
            )
            frontiers.extend([(node, level - 1) for node in nodes])
            edges_list.extend(edges)

    if not visited_nodes:
        return ""

    # Create incident-based representation
    rtn_strs = []
    for node in visited_nodes:
        if node not in code_graph.name_to_vertex:
            continue

        vertex_id = code_graph.name_to_vertex[node]
        vertex = code_graph.graph.vs[vertex_id]
        ntype = vertex.get("type", "unknown")

        incident_dict = defaultdict(lambda: defaultdict(list))

        # Find all edges involving this node
        for edge in edges_list:
            if edge[0] == node:  # outgoing edge
                target = edge[1]
                etype = edge[3]["type"]
                target_vertex_id = code_graph.name_to_vertex.get(target)
                if target_vertex_id is not None:
                    target_type = code_graph.graph.vs[target_vertex_id].get(
                        "type", "unknown"
                    )
                    incident_dict[etype][target_type].append(target)

        if incident_dict:
            s = f'{ntype} "{node}" '
            for edge_type, neigh_dict in incident_dict.items():
                s += f"{edge_type} "
                for neigh_type, neigh_list in neigh_dict.items():
                    neigh_list = ['"' + neigh + '"' for neigh in neigh_list]
                    s += f'{neigh_type} {",".join(neigh_list)} '
                s = s[:-1] + "; and "
            s = s[:-6] + "."
            rtn_strs.append(s)

    return "\n\n".join(rtn_strs)


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
        return f"Node '{root}' not found in graph"

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

        neigh_ids, etypes, edirs = [], [], []

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
                neigh_type = neighbor_vertex.get("type", "unknown")

                if is_ntype_not_valid(neigh_type):
                    continue

                # Get edge type
                edge_id = code_graph.graph.get_eid(vertex_id, neighbor_id, error=False)
                if edge_id is not None:
                    etype = code_graph.graph.es[edge_id].get("type", "unknown")
                    if is_etype_not_valid(etype):
                        continue
                    if not is_test_file(neighbor):
                        if (node, etype, neighbor) not in traversed_edges:
                            neigh_ids.append(neighbor)
                            etypes.append(etype)
                            edirs.append("downstream")
                            traversed_edges.add((node, etype, neighbor))

        # Upstream traversal
        if "upstream" == edirection or (node == root and direction == "both"):
            neighbor_ids = code_graph.graph.predecessors(vertex_id)
            for neighbor_id in neighbor_ids:
                neighbor_vertex = code_graph.graph.vs[neighbor_id]
                neighbor = neighbor_vertex["name"]
                neigh_type = neighbor_vertex.get("type", "unknown")

                if is_ntype_not_valid(neigh_type):
                    continue

                # Get edge type
                edge_id = code_graph.graph.get_eid(neighbor_id, vertex_id, error=False)
                if edge_id is not None:
                    etype = code_graph.graph.es[edge_id].get("type", "unknown")
                    if is_etype_not_valid(etype):
                        continue
                    if not is_test_file(neighbor):
                        if (neighbor, etype, node) not in traversed_edges:
                            neigh_ids.append(neighbor)
                            etypes.append(etype)
                            edirs.append("upstream")
                            traversed_edges.add((neighbor, etype, node))

        for i, (neigh_id, etype, edir) in enumerate(zip(neigh_ids, etypes, edirs)):
            is_last_child = i == len(neigh_ids) - 1
            if edir == "upstream":
                etype += "-by"
            traverse(neigh_id, new_prefix, is_last_child, level + 1, etype, edir)

    traverse(root, "", False, 0, None, None)
    return "\n".join(rtn_str)
