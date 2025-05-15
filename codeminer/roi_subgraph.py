from collections import deque
from typing import Any, Dict, List, Optional, Set

import igraph as ig

from .code_graph import CodeGraph
from .log_utils import get_logger

logger = get_logger(__name__)


class ROISubgraph:
    """
    Region of Interest (ROI) subgraph extraction for code graphs.

    This class takes search results from CodeSearchEngine and extracts a focused
    subgraph containing the most relevant code elements and their neighborhood.
    """

    def __init__(self, code_graph: CodeGraph):
        """
        Initialize the ROI subgraph extractor.

        Args:
            code_graph: The full code graph from which to extract subgraphs
        """
        self.code_graph = code_graph
        self.full_graph = code_graph.get_graph()

    def extract_subgraph(
        self,
        node_ids: List[int],
        k_hop: int = 2,
        edge_types: Optional[List[str]] = None,
    ) -> ig.Graph:
        """
        Extract a subgraph around specified node IDs.

        Args:
            node_ids: List of node IDs to use as seeds for the subgraph
            k_hop: Number of hops to expand from each seed node
            edge_types: Optional list of edge types to consider (None for all types)

        Returns:
            An igraph Graph object representing the extracted subgraph
        """
        # Track visited nodes to avoid duplicates
        visited_nodes = set()
        # Track nodes to include in the subgraph
        subgraph_nodes = set()

        # BFS for each seed node
        for node_id in node_ids:
            try:
                # Skip if we've already processed this node
                if node_id in visited_nodes:
                    continue

                # Add this seed node to the subgraph and mark as visited
                subgraph_nodes.add(node_id)
                visited_nodes.add(node_id)

                # Initialize queue for BFS with (node, depth) pairs
                queue = deque([(node_id, 0)])

                # BFS traversal to find k-hop neighborhood
                while queue:
                    current_node, depth = queue.popleft()

                    # Stop expanding if we've reached k hops
                    if depth >= k_hop:
                        continue

                    # Get neighbors
                    neighbors = self._get_neighbors(current_node, edge_types)

                    # Process neighbors
                    for neighbor in neighbors:
                        if neighbor not in visited_nodes:
                            visited_nodes.add(neighbor)
                            subgraph_nodes.add(neighbor)
                            queue.append((neighbor, depth + 1))

            except Exception as e:
                logger.error(f"Error processing node {node_id}: {e}")
                continue

        # Create subgraph from the collected nodes
        return self._create_subgraph(subgraph_nodes)

    def extract_roi_from_search_results(
        self,
        search_results: List[Dict[str, Any]],
        k_hop: int = 2,
        max_nodes: int = 20,
        edge_types: Optional[List[str]] = None,
    ) -> ig.Graph:
        """
        Extract a region of interest subgraph from search results.

        Args:
            search_results: Search results from BM25CodeIndexer.search()
            k_hop: Number of hops to explore from each seed node
            max_nodes: Maximum number of seed nodes to use (to limit subgraph size)
            edge_types: Optional list of edge types to include (None for all types)

        Returns:
            An igraph Graph object representing the ROI subgraph
        """
        # Extract node_ids from search results
        node_ids = []
        for result in search_results[:max_nodes]:  # Limit the number of seed nodes
            if "node_id" in result:
                node_ids.append(result["node_id"])

        # If no valid node IDs found, return an empty graph
        if not node_ids:
            logger.warning("No valid node IDs found in search results.")
            return ig.Graph()

        logger.info(
            f"Extracting ROI subgraph with {len(node_ids)} seed nodes and {k_hop} hops"
        )
        return self.extract_subgraph(node_ids, k_hop, edge_types)

    def get_filtered_subgraph_nodes(
        self,
        subgraph: ig.Graph,
        node_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract useful nodes from a subgraph, filtering out nodes where
        start_line equals end_line (unless explicitly included).

        Args:
            subgraph: An igraph Graph object (from extract_subgraph or extract_roi_from_search_results)
            node_types: Optional list of node types to include (None for all types)

        Returns:
            List of dictionaries containing node attributes for useful nodes
        """
        filtered_nodes = []

        # Process each node in the subgraph
        for node in subgraph.vs:
            try:
                # Get original node ID and attributes
                original_id = node["original_id"]
                node_attrs = self.get_node_info(original_id)

                # Check if this node has start_line and end_line attributes
                has_line_info = "start_line" in node_attrs and "end_line" in node_attrs

                # Filter by node type if specified
                if node_types and "type" in node_attrs:
                    if node_attrs["type"] not in node_types:
                        continue

                # Skip nodes with zero line span unless explicitly included
                if has_line_info:
                    if node_attrs["start_line"] == node_attrs["end_line"]:
                        continue

                # Add original_id to the node attributes
                node_attrs["original_id"] = original_id

                # Add to filtered nodes list
                filtered_nodes.append(node_attrs)

            except Exception as e:
                logger.error(f"Error processing node {node.index}: {e}")
                continue

        logger.info(f"Extracted {len(filtered_nodes)} useful nodes from subgraph")
        return filtered_nodes

    def _get_neighbors(
        self, node_id: int, edge_types: Optional[List[str]] = None
    ) -> List[int]:
        """
        Get neighbors of a node, optionally filtered by edge type.

        Args:
            node_id: ID of the node
            edge_types: Optional list of edge types to filter by

        Returns:
            List of neighbor node IDs
        """
        # Get all neighbor IDs
        neighbors = self.full_graph.neighbors(node_id)

        # If no edge type filter, return all neighbors
        if not edge_types:
            return neighbors

        # Filter neighbors by edge type
        filtered_neighbors = []
        for neighbor in neighbors:
            # Get edge between node and neighbor
            try:
                edge_id = self.full_graph.get_eid(node_id, neighbor, error=False)
                if edge_id is None:
                    # Try the reverse direction
                    edge_id = self.full_graph.get_eid(neighbor, node_id, error=False)

                if (
                    edge_id is not None
                    and "type" in self.full_graph.es[edge_id].attributes()
                ):
                    edge_type = self.full_graph.es[edge_id]["type"]
                    if edge_type in edge_types:
                        filtered_neighbors.append(neighbor)
            except Exception as e:
                logger.error(f"Error getting edge type: {e}")
                continue

        return filtered_neighbors

    def _create_subgraph(self, node_ids: Set[int]) -> ig.Graph:
        """
        Create a subgraph from a set of node IDs.

        Args:
            node_ids: Set of node IDs to include in the subgraph

        Returns:
            An igraph Graph object representing the subgraph
        """
        # Convert set to list to have stable indices
        node_list = list(node_ids)

        # Create a subgraph from the original graph
        subgraph = self.full_graph.subgraph(node_list)

        # Add original node IDs as an attribute to make mapping back easier
        for i, original_id in enumerate(node_list):
            subgraph.vs[i]["original_id"] = original_id

        return subgraph

    def get_node_info(self, node_id: int) -> Dict[str, Any]:
        """
        Get information about a node in the original graph.

        Args:
            node_id: ID of the node in the original graph

        Returns:
            Dictionary of node attributes
        """
        try:
            return self.full_graph.vs[node_id].attributes()
        except Exception as e:
            logger.error(f"Error getting node info for node {node_id}: {e}")
            return {}
