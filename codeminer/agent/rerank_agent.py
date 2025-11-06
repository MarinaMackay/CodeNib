"""
Rerank agent for ranking code nodes based on relevance to a query.
This implementation delegates ranking to the rank_llm Reranker abstraction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from rank_llm.data import Candidate as RankLLMCandidate
from rank_llm.data import Query as RankLLMQuery
from rank_llm.data import Request as RankLLMRequest
from rank_llm.data import Result as RankLLMResult
from rank_llm.rerank import Reranker as RankLLMReranker

from ..llm.llm_config import LLMConfig
from ..log_utils import get_logger
from ..types import NodeWithContent, NodeWithScore

logger = get_logger(__name__)


class RerankResult(BaseModel):
    """Model kept for backwards compatibility with previous API."""

    ranked_indices: List[int] = Field(
        description="List of node indices ranked by relevance (most relevant first)"
    )
    scores: List[float] = Field(
        description="Relevance scores for each ranked node (0.0 to 1.0)"
    )


class RerankAgent:
    """Agent for reranking code nodes using the rank_llm Reranker."""

    def __init__(
        self,
        llm_config: LLMConfig,
        *,
        reranker_kwargs: Optional[Dict[str, Any]] = None,
        interactive: bool = False,
        **kwargs: Any,
    ):
        """Initialize the rerank agent.

        Args:
            llm_config: Configuration describing which rerank model to use.
            reranker_kwargs: Extra keyword arguments forwarded to the rank_llm factory.
            interactive: Whether to construct the RankLLM coordinator in interactive mode.
            **kwargs: Additional keyword arguments forwarded to the rank_llm factory.
        """

        self.llm_config = llm_config
        self._candidate_cls = RankLLMCandidate
        self._request_cls = RankLLMRequest
        self._request_counter = 0

        aggregated_kwargs = self._build_reranker_kwargs(
            llm_config=llm_config,
            overrides=reranker_kwargs,
            extra_kwargs=kwargs,
        )

        try:
            model_coordinator = RankLLMReranker.create_model_coordinator(
                model_path=llm_config.model_name,
                default_model_coordinator=None,
                interactive=interactive,
                **aggregated_kwargs,
            )
        except Exception as exc:  # pragma: no cover - depends on external package
            raise RuntimeError(
                f"Failed to construct RankLLM coordinator for model "
                f"{llm_config.model_name}: {exc}"
            ) from exc

        if model_coordinator is None:
            raise RuntimeError(
                f"RankLLM returned no model coordinator for {llm_config.model_name}"
            )
        self.reranker = RankLLMReranker(model_coordinator=model_coordinator)

        logger.info(
            "Initialized rerank agent with RankLLM model: %s",
            self.llm_config.model_name,
        )

    def rerank_nodes(
        self, query: str, nodes: List[NodeWithContent], top_k: Optional[int] = None
    ) -> List[NodeWithScore]:
        """
        Rerank nodes based on their relevance to the query.

        Args:
            query: The query to rank nodes against.
            nodes: List of nodes with content to rank.
            top_k: Optional cap on the number of ranked results to return.

        Returns:
            List of nodes ranked according to the external reranker.
        """
        if not nodes:
            logger.warning("No nodes provided for reranking.")
            return []

        if not query or not query.strip():
            logger.warning("Empty query provided for reranking.")
            return []

        valid_nodes: List[Tuple[int, NodeWithContent]] = [
            (idx, node)
            for idx, node in enumerate(nodes)
            if node.content and node.content.strip()
        ]

        if not valid_nodes:
            logger.warning("No nodes with content found for reranking.")
            return []

        doc_map: Dict[str, NodeWithContent] = {}
        candidates: List[Any] = []
        for position, (original_index, node) in enumerate(valid_nodes):
            doc_id = self._make_doc_id(node, original_index, position)
            doc_map[doc_id] = node
            metadata = {
                "original_index": original_index,
                "position": position,
                "node_name": node.node_name,
                "file": node.file,
                "type": node.type,
                "start_line": node.start_line,
                "end_line": node.end_line,
            }
            candidate = self._build_candidate(
                doc_id=doc_id,
                text=node.content,
                metadata=metadata,
            )
            candidates.append(candidate)

        request = self._build_request(query=query, candidates=candidates)
        rank_limit = len(candidates)
        if top_k is not None:
            rank_limit = max(1, min(top_k, rank_limit))

        logger.info(
            "Reranking %d nodes via rank_llm (limit=%d).", len(candidates), rank_limit
        )

        try:
            result = self.reranker.rerank(
                request=request,
                rank_start=0,
                rank_end=rank_limit,
                shuffle_candidates=False,
            )
        except Exception as exc:  # pragma: no cover - depends on external package
            logger.error("RankLLM rerank failed: %s", exc)
            return []

        reranked_candidates = self._extract_reranked(result)
        if not reranked_candidates:
            logger.warning("RankLLM returned no reranked candidates.")
            return []

        max_results = min(len(reranked_candidates), rank_limit)
        ranked_nodes: List[NodeWithScore] = []

        for candidate in reranked_candidates[:max_results]:
            doc_id = self._extract_doc_id(candidate)
            if doc_id is None:
                continue
            node = doc_map.get(doc_id)
            if node is None:
                continue
            score = self._extract_score(candidate)
            ranked_nodes.append(
                NodeWithScore(
                    node_name=node.node_name,
                    type=node.type,
                    file=node.file,
                    start_line=node.start_line,
                    end_line=node.end_line,
                    score=score,
                )
            )

        logger.info("Successfully reranked %d nodes.", len(ranked_nodes))
        return ranked_nodes

    def rerank_with_metadata(
        self,
        query: str,
        nodes: List[NodeWithContent],
        top_k: Optional[int] = None,
        include_content: bool = False,
    ) -> List[dict]:
        """
        Rerank nodes and return detailed results with metadata.
        """
        ranked_nodes = self.rerank_nodes(query, nodes, top_k)

        results: List[dict] = []
        content_lookup = {node.node_name: node for node in nodes if node.node_name}
        for idx, node in enumerate(ranked_nodes):
            payload = {
                "rank": idx + 1,
                "score": node.score,
                "node_name": node.node_name,
                "type": node.type,
                "file": node.file,
                "start_line": node.start_line,
                "end_line": node.end_line,
            }
            if include_content:
                origin = content_lookup.get(node.node_name)
                if origin and origin.content:
                    payload["content"] = origin.content
                    payload["content_preview"] = (
                        origin.content[:150] + "..."
                        if len(origin.content) > 150
                        else origin.content
                    )
            results.append(payload)
        return results

    def _build_candidate(
        self,
        *,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RankLLMCandidate:
        metadata = dict(metadata or {})
        metadata.setdefault("doc_id", doc_id)
        score = float(metadata.get("score", 0.0))
        return self._candidate_cls(
            docid=doc_id,
            score=score,
            doc={
                "text": text,
                "metadata": metadata,
            },
        )

    def _build_request(
        self, *, query: str, candidates: List[RankLLMCandidate]
    ) -> RankLLMRequest:
        self._request_counter += 1
        request_id = f"codeminer-{self._request_counter}"
        query_payload = RankLLMQuery(text=query, qid=request_id)
        return self._request_cls(query=query_payload, candidates=candidates)

    @staticmethod
    def _make_doc_id(node: NodeWithContent, original_index: int, position: int) -> str:
        suffix = node.node_name or f"node-{original_index}"
        return f"{original_index}-{position}-{suffix}"

    @staticmethod
    def _extract_reranked(result: RankLLMResult) -> List[RankLLMCandidate]:
        return list(result.candidates or [])

    def _extract_doc_id(self, candidate: RankLLMCandidate) -> Optional[str]:
        if candidate.docid not in (None, ""):
            return str(candidate.docid)
        doc_payload = candidate.doc if isinstance(candidate.doc, dict) else {}
        metadata = (
            doc_payload.get("metadata") if isinstance(doc_payload, dict) else None
        )
        if isinstance(metadata, dict):
            for key in ("doc_id", "docid", "document_id", "id"):
                value = metadata.get(key)
                if value not in (None, ""):
                    return str(value)
        return None

    @staticmethod
    def _extract_score(candidate: RankLLMCandidate) -> float:
        if isinstance(candidate.score, (int, float)):
            return float(candidate.score)
        doc_payload = candidate.doc if isinstance(candidate.doc, dict) else {}
        metadata = (
            doc_payload.get("metadata") if isinstance(doc_payload, dict) else None
        )
        if isinstance(metadata, dict):
            for key in ("score", "rerank_score", "relevance_score"):
                value = metadata.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        return 0.0

    def _build_reranker_kwargs(
        self,
        *,
        llm_config: LLMConfig,
        overrides: Optional[Dict[str, Any]],
        extra_kwargs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        aggregated: Dict[str, Any] = {}
        config_data = llm_config.config_data or {}

        reranker_config = config_data.get("reranker_kwargs")
        if isinstance(reranker_config, dict):
            aggregated.update(reranker_config)

        if overrides:
            aggregated.update(overrides)
        if extra_kwargs:
            aggregated.update(extra_kwargs)

        # Surface commonly used configuration entries to rank_llm.
        base_url_env = llm_config.get_config_value_optional("OPENAI_API_BASE_URL")
        base_url_sources = [
            config_data.get("OPENAI_API_BASE_URL"),
            config_data.get("base_url"),
            aggregated.get("base_url"),
            base_url_env,
        ]
        for url in base_url_sources:
            if isinstance(url, str) and url.strip():
                aggregated.setdefault("base_url", url.strip())
                break

        # Detect OpenRouter usage from config or environment.
        if "use_openrouter" not in aggregated:
            base_url_val = aggregated.get("base_url") or base_url_env
            if isinstance(base_url_val, str) and "openrouter" in base_url_val.lower():
                aggregated["use_openrouter"] = True
                aggregated.setdefault("base_url", base_url_val)
            else:
                openrouter_key = llm_config.get_config_value_optional(
                    "OPENROUTER_API_KEY"
                )
                if openrouter_key:
                    aggregated["use_openrouter"] = True
                    aggregated.setdefault("base_url", "https://openrouter.ai/api/v1/")

        # Propagate Azure OpenAI usage if credentials are configured.
        if "use_azure_openai" not in aggregated:
            azure_key = llm_config.get_config_value_optional("AZURE_OPENAI_API_KEY")
            if azure_key:
                aggregated["use_azure_openai"] = True

        return aggregated


def rerank_nodes_with_query(
    query: str,
    nodes: List[NodeWithContent],
    llm_config: LLMConfig,
    top_k: Optional[int] = None,
    reranker_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[NodeWithScore]:
    """
    Convenience function to rerank nodes with a query.
    """
    agent = RerankAgent(
        llm_config=llm_config,
        reranker_kwargs=reranker_kwargs,
        **kwargs,
    )
    return agent.rerank_nodes(query, nodes, top_k)
