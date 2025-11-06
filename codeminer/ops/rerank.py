from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..agent.rerank_agent import RerankAgent
from ..llm.llm_config import LLMConfig
from ..log_utils import get_logger
from ..plans.execution import ExecutionEngine
from ..plans.ir_exec import ExecutionNode
from ..plans.ir_physical import PhysicalOperator
from ..types import NodeWithContent, NodeWithScore, NodeWithScoreContent

logger = get_logger(__name__)


@dataclass
class RerankContext:
    """Shared context carrying the rerank agent and configuration."""

    llm_config: Optional[LLMConfig] = None
    agent: Optional[RerankAgent] = None
    top_k: Optional[int] = None

    def ensure_agent(self) -> RerankAgent:
        if self.agent is None:
            if self.llm_config is None:
                raise RuntimeError("Rerank agent requested but no LLMConfig provided.")
            logger.info(
                "Creating rerank agent.",
                extra={"model": self.llm_config.model_name},
            )
            self.agent = RerankAgent(llm_config=self.llm_config)
        return self.agent


def register_rerank_ops(engine: ExecutionEngine, context: RerankContext) -> None:
    """Register rerank kernels on the execution engine."""

    engine.register(PhysicalOperator.LLM_RERANK.value, _llm_rerank_kernel(context))


def _llm_rerank_kernel(context: RerankContext):
    def run(node: ExecutionNode, inputs: List[object]) -> List[NodeWithScoreContent]:
        query = _extract_query(node.params, inputs)
        candidates = _collect_candidates(inputs)
        if not candidates:
            logger.warning("LLM rerank invoked without candidate nodes.")
            return []

        top_k = _resolve_top_k(node.params, context.top_k)
        include_content = bool(node.params.get("return_content", False))

        agent = context.ensure_agent()
        logger.info(
            "Running LLM rerank.",
            extra={"candidate_count": len(candidates), "top_k": top_k},
        )

        ranked = agent.rerank_nodes(
            query=query,
            nodes=candidates,
            top_k=top_k,
        )

        if include_content:
            return _attach_content(ranked, candidates)
        return _to_score_content(ranked)

    return run


def _extract_query(params: Dict[str, object], inputs: List[object]) -> str:
    for key in ("query", "prompt", "text"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for payload in inputs:
        if isinstance(payload, dict):
            value = payload.get("query") or payload.get("source_query")
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ValueError("Rerank operator requires the original query text.")


def _collect_candidates(inputs: List[object]) -> List[NodeWithContent]:
    collected: List[NodeWithContent] = []
    for payload in inputs:
        if isinstance(payload, NodeWithContent):
            collected.append(payload)
            continue
        if isinstance(payload, NodeWithScoreContent):
            collected.append(
                NodeWithContent(
                    node_name=payload.node_name,
                    type=payload.type,
                    file=payload.file,
                    start_line=payload.start_line,
                    end_line=payload.end_line,
                    content=payload.content,
                )
            )
            continue
        if isinstance(payload, dict):
            nodes = (
                payload.get("nodes")
                or payload.get("results")
                or payload.get("candidates")
            )
            if nodes:
                collected.extend(_collect_candidates(list(nodes)))
            continue
        if isinstance(payload, list):
            collected.extend(_collect_candidates(payload))
    return collected


def _resolve_top_k(params: Dict[str, object], default: Optional[int]) -> Optional[int]:
    for key in ("top_k", "limit"):
        value = params.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return default


def _attach_content(
    ranked: List[NodeWithScore], candidates: List[NodeWithContent]
) -> List[NodeWithScoreContent]:
    content_index: Dict[str, NodeWithContent] = {
        c.node_name: c for c in candidates if c.node_name
    }
    enriched: List[NodeWithScoreContent] = []
    for node in ranked:
        content = content_index.get(node.node_name)
        payload = node.model_dump()
        payload["content"] = content.content if content else None
        enriched.append(NodeWithScoreContent(**payload))
    return enriched


def _to_score_content(results: List[NodeWithScore]) -> List[NodeWithScoreContent]:
    return [NodeWithScoreContent(**node.model_dump()) for node in results]
