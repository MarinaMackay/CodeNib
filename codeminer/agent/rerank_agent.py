"""
Rerank agent for ranking code nodes based on relevance to a query.
This module uses LLM APIs to rank NodeInfo objects and return QueriedNode objects.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from ..llm.litellm_chat import LiteLLMChat, human_message, system_message
from ..log_utils import get_logger
from ..types import NodeInfo, QueriedNode

logger = get_logger(__name__)


# Define Pydantic model for structured output
class RerankResult(BaseModel):
    """Model for reranking output."""

    ranked_indices: List[int] = Field(
        description="List of node indices ranked by relevance (most relevant first)"
    )
    scores: List[float] = Field(
        description="Relevance scores for each ranked node (0.0 to 1.0)"
    )


class RerankAgent:
    """Agent for reranking code nodes based on query relevance using LLM APIs."""

    def __init__(
        self,
        llm: LiteLLMChat,
    ):
        """Initialize the rerank agent.

        Args:
            llm: A configured LiteLLMChat instance.
        """
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(RerankResult)
        logger.info(f"Initialized rerank agent with model: {self.llm.model}")

    def rerank_nodes(
        self,
        query: str,
        nodes: List[NodeInfo],
        top_k: Optional[int] = None,
        window_size: Optional[int] = None,
        window_step: Optional[int] = None,
        include_content: bool = False,
    ) -> List[QueriedNode]:
        """
        Rerank nodes based on their relevance to the query.

        Args:
            query (str): The query to rank nodes against
            nodes (List[NodeInfo]): List of nodes with content to rank
            top_k (Optional[int]): Maximum number of results to return (None for all)
            window_size (Optional[int]): Number of nodes per rerank window. None -> all nodes.
            window_step (Optional[int]): Step size between sliding
                windows. Defaults to window_size.
            include_content (bool): Whether to include node content in the result objects.

        Returns:
            List[QueriedNode]: Ranked nodes with relevance scores (optionally with content)

        Notes:
            When a sliding window is configured, each window is reranked independently and
            the averaged scores across all windows determine the final ordering.
        """
        if not nodes:
            logger.warning("No nodes provided for reranking")
            return []

        if not query.strip():
            logger.warning("Empty query provided for reranking")
            return []

        try:
            # Filter nodes with content
            valid_nodes: List[Tuple[int, NodeInfo]] = []
            for i, node in enumerate(nodes):
                if node.content and node.content.strip():
                    valid_nodes.append((i, node))

            if not valid_nodes:
                logger.warning("No nodes with content found for reranking")
                return []

            logger.info(
                f"Reranking {len(valid_nodes)} nodes with query: {query[:100]}..."
            )

            node_lookup: Dict[int, NodeInfo] = {
                original_idx: node for original_idx, node in valid_nodes
            }

            total_nodes = len(valid_nodes)
            window_size = window_size or total_nodes
            if window_size <= 0:
                window_size = total_nodes

            window_step = window_step or window_size
            if window_step <= 0:
                window_step = window_size

            aggregated_scores: Dict[int, float] = defaultdict(float)
            appearance_count: Counter[int] = Counter()

            window_starts = list(range(0, total_nodes, window_step))
            tail_start = max(total_nodes - window_size, 0)
            if tail_start not in window_starts:
                window_starts.append(tail_start)
            window_starts = sorted(set(window_starts))

            window_id = 0
            for window_start in window_starts:
                window_nodes = valid_nodes[window_start : window_start + window_size]
                if not window_nodes:
                    continue

                window_id += 1
                logger.debug(
                    "Processing rerank window %s (nodes %s-%s)",
                    window_id,
                    window_start,
                    window_start + len(window_nodes) - 1,
                )

                window_results = self._rerank_window(query, window_nodes)
                for original_idx, score in window_results:
                    aggregated_scores[original_idx] += float(score)
                    appearance_count[original_idx] += 1

            if not aggregated_scores:
                logger.warning("Reranker did not return any scores.")
                return []

            averaged_scores = {
                idx: aggregated_scores[idx] / appearance_count[idx]
                for idx in aggregated_scores
                if appearance_count[idx] > 0
            }

            sorted_indices = sorted(
                averaged_scores.items(), key=lambda item: item[1], reverse=True
            )

            ranked_nodes: List[QueriedNode] = []
            for original_idx, score in sorted_indices:
                node = node_lookup.get(original_idx)
                if not node:
                    continue
                payload = {
                    "node_name": node.node_name,
                    "type": node.type,
                    "file": node.file,
                    "node_id": node.node_id,
                    "start_line": node.start_line,
                    "end_line": node.end_line,
                    "score": float(score),
                }
                if include_content:
                    payload["content"] = node.content
                ranked_nodes.append(QueriedNode(**payload))
                if top_k and len(ranked_nodes) >= top_k:
                    break

            logger.info(
                "Successfully reranked %s nodes across %s window(s)",
                len(ranked_nodes),
                window_id or 1,
            )
            return ranked_nodes

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return []

    def _rerank_window(
        self, query: str, window_nodes: Sequence[Tuple[int, NodeInfo]]
    ) -> List[Tuple[int, float]]:
        """Invoke the LLM to rerank a specific window of nodes."""
        if not window_nodes:
            return []

        system_prompt = (
            "You are a code relevance ranking specialist. Your task is to rank code snippets "
            "based on their relevance to a given query. You should consider:\n\n"
            "1. Semantic relevance - how well the code addresses the query concepts\n"
            "2. Functional relevance - whether the code implements related functionality\n"
            "3. Contextual relevance - how the code fits within the broader context\n"
            "4. Technical accuracy - whether the code is technically sound for the query\n\n"
            "Return the indices of nodes ranked from most relevant to least relevant, "
            "along with relevance scores between 0.0 and 1.0."
        )

        node_descriptions = []
        for i, (_, node) in enumerate(window_nodes):
            description = (
                f"Node {i}:\n"
                f"Name: {node.node_name}\n"
                f"Type: {node.type}\n"
                f"File: {node.file}\n"
                f"Lines: {node.start_line}-{node.end_line}\n"
                f"Content:\n{node.content[:1000]}"
                f"{'...' if len(node.content) > 1000 else ''}\n\n"
            )
            node_descriptions.append(description)

        user_prompt = (
            f"Query: {query}\n\n"
            f"Please rank the following {len(window_nodes)} "
            "code nodes by relevance to the query:\n\n"
            + "\n".join(node_descriptions)
            + f"\n\nRank all {len(window_nodes)} nodes by relevance and provide scores. "
            "Return the indices (0-based) in order of relevance (most relevant first) "
            "and corresponding relevance scores (0.0 to 1.0)."
        )

        system_msg = system_message(system_prompt)
        user_msg = human_message(user_prompt)

        try:
            rerank_result = self.structured_llm.invoke([system_msg, user_msg])
        except Exception as exc:
            logger.error("LLM rerank invocation failed: %s", exc)
            return []

        window_scores: List[Tuple[int, float]] = []
        for local_position, ranked_idx in enumerate(rerank_result.ranked_indices):
            if local_position >= len(rerank_result.scores):
                break
            if not 0 <= ranked_idx < len(window_nodes):
                continue
            score = rerank_result.scores[local_position]
            original_idx = window_nodes[ranked_idx][0]
            window_scores.append((original_idx, float(score)))

        logger.debug(
            "Window rerank produced %s scored nodes (window size %s)",
            len(window_scores),
            len(window_nodes),
        )
        return window_scores


def rerank_nodes_with_query(
    query: str,
    nodes: List[NodeInfo],
    llm: LiteLLMChat,
    top_k: Optional[int] = None,
    window_size: Optional[int] = None,
    window_step: Optional[int] = None,
    include_content: bool = False,
) -> List[QueriedNode]:
    """
    Convenience function to rerank nodes with a query.

    Args:
        query: The query to rank nodes against.
        nodes: List of nodes with content to rank.
        llm: A configured LiteLLMChat instance.
        top_k: Maximum number of results to return (None for all).
        window_size: Optional sliding window size for reranking.
        window_step: Optional stride between sliding windows.
        include_content: Whether to include node content in the result objects.

    Returns:
        List[QueriedNode]: Ranked nodes with relevance scores
    """
    agent = RerankAgent(llm=llm)
    return agent.rerank_nodes(
        query,
        nodes,
        top_k=top_k,
        window_size=window_size,
        window_step=window_step,
        include_content=include_content,
    )
