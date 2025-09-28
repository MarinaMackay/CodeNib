"""
Rerank agent for ranking code nodes based on relevance to a query.
This module uses LLM APIs to rank NodeWithContent objects and return NodeWithScore objects.
"""

from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..llm.llm_config import LLMConfig, create_llm
from ..log_utils import get_logger
from ..types import NodeWithContent, NodeWithScore

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
        llm_config: LLMConfig,
        **kwargs,
    ):
        """Initialize the rerank agent.

        Args:
            llm_config: LLM configuration containing model, provider, and other settings.
            **kwargs: Additional keyword arguments forwarded to the LLM factory.
        """
        self.llm_config = llm_config
        self.llm = create_llm(
            config=llm_config,
            **kwargs,
        )

        # Convert the LLM to a structured output LLM
        self.structured_llm = self.llm.with_structured_output(RerankResult)
        logger.info(
            f"Initialized rerank agent with model: {self.llm_config.model_name}"
        )

    def rerank_nodes(
        self, query: str, nodes: List[NodeWithContent], top_k: Optional[int] = None
    ) -> List[NodeWithScore]:
        """
        Rerank nodes based on their relevance to the query.

        Args:
            query (str): The query to rank nodes against
            nodes (List[NodeWithContent]): List of nodes with content to rank
            top_k (Optional[int]): Maximum number of results to return (None for all)

        Returns:
            List[NodeWithScore]: Ranked nodes with relevance scores
        """
        if not nodes:
            logger.warning("No nodes provided for reranking")
            return []

        if not query.strip():
            logger.warning("Empty query provided for reranking")
            return []

        try:
            # Filter nodes with content
            valid_nodes = []
            for i, node in enumerate(nodes):
                if node.content and node.content.strip():
                    valid_nodes.append((i, node))

            if not valid_nodes:
                logger.warning("No nodes with content found for reranking")
                return []

            logger.info(
                f"Reranking {len(valid_nodes)} nodes with query: {query[:100]}..."
            )

            # Create prompt with system instructions and node information
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

            # Prepare node descriptions for ranking
            node_descriptions = []
            for i, (_, node) in enumerate(valid_nodes):
                description = (
                    f"Node {i}:\n"
                    f"Name: {node.node_name}\n"
                    f"Type: {node.type}\n"
                    f"File: {node.file}\n"
                    f"Lines: {node.start_line}-{node.end_line}\n"
                    f"Content:\n{node.content[:1000]}{'...' if len(node.content) > 1000 else ''}\n\n"
                )
                node_descriptions.append(description)

            user_prompt = (
                f"Query: {query}\n\n"
                f"Please rank the following {len(valid_nodes)} code nodes by relevance to the query:\n\n"
                + "\n".join(node_descriptions)
                + f"\n\nRank all {len(valid_nodes)} nodes by relevance and provide scores. "
                "Return the indices (0-based) in order of relevance (most relevant first) "
                "and corresponding relevance scores (0.0 to 1.0)."
            )

            # Use structured LLM to get ranking
            system_msg = SystemMessage(content=system_prompt)
            user_msg = HumanMessage(content=user_prompt)
            result = self.structured_llm.invoke([system_msg, user_msg])

            rerank_result = result
            logger.debug(f"Rerank result: {rerank_result}")

            # Create ranked results with node information
            ranked_nodes = []
            max_results = (
                min(len(rerank_result.ranked_indices), top_k)
                if top_k
                else len(rerank_result.ranked_indices)
            )

            for i in range(max_results):
                if i < len(rerank_result.ranked_indices) and i < len(
                    rerank_result.scores
                ):
                    ranked_idx = rerank_result.ranked_indices[i]
                    score = rerank_result.scores[i]

                    if 0 <= ranked_idx < len(valid_nodes):
                        original_node = valid_nodes[ranked_idx][1]

                        # Create NodeWithScore
                        node_with_score = NodeWithScore(
                            node_name=original_node.node_name,
                            type=original_node.type,
                            file=original_node.file,
                            start_line=original_node.start_line,
                            end_line=original_node.end_line,
                            score=float(score),
                        )

                        ranked_nodes.append(node_with_score)

            logger.info(f"Successfully reranked {len(ranked_nodes)} nodes")
            return ranked_nodes

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return []

    def rerank_with_metadata(
        self,
        query: str,
        nodes: List[NodeWithContent],
        top_k: Optional[int] = None,
        include_content: bool = False,
    ) -> List[dict]:
        """
        Rerank nodes and return detailed results with metadata.

        Args:
            query (str): The query to rank nodes against
            nodes (List[NodeWithContent]): List of nodes with content to rank
            top_k (Optional[int]): Maximum number of results to return (None for all)
            include_content (bool): Whether to include content in the results

        Returns:
            List[dict]: Ranked results with detailed metadata
        """
        ranked_nodes = self.rerank_nodes(query, nodes, top_k)

        results = []
        for i, node in enumerate(ranked_nodes):
            result = {
                "rank": i + 1,
                "score": node.score,
                "node_name": node.node_name,
                "type": node.type,
                "file": node.file,
                "start_line": node.start_line,
                "end_line": node.end_line,
            }

            if include_content:
                # Find the original node to get content
                original_node = next(
                    (n for n in nodes if n.node_name == node.node_name), None
                )
                if original_node:
                    result["content"] = original_node.content
                    result["content_preview"] = (
                        original_node.content[:150] + "..."
                        if len(original_node.content) > 150
                        else original_node.content
                    )

            results.append(result)

        return results


def rerank_nodes_with_query(
    query: str,
    nodes: List[NodeWithContent],
    llm_config: LLMConfig,
    top_k: Optional[int] = None,
    **kwargs,
) -> List[NodeWithScore]:
    """
    Convenience function to rerank nodes with a query.

    Args:
        query (str): The query to rank nodes against
        nodes (List[NodeWithContent]): List of nodes with content to rank
        llm_config (LLMConfig): LLM configuration containing model, provider, and other settings.
        top_k (Optional[int]): Maximum number of results to return (None for all)
        **kwargs: Additional arguments to pass to the LLM

    Returns:
        List[NodeWithScore]: Ranked nodes with relevance scores
    """
    agent = RerankAgent(
        llm_config=llm_config,
        **kwargs,
    )
    return agent.rerank_nodes(query, nodes, top_k)
