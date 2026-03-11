from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..agent.extract_agent import KeywordExtractor
from ..compiler.execution import ExecutionEngine
from ..compiler.ir_exec import ExecutionNode
from ..compiler.ir_physical import PhysicalOperator
from ..llm.llm_config import LLMConfig
from ..log_utils import get_logger
from ..types import QueriedNode

logger = get_logger(__name__)


@dataclass
class TransformContext:
    """Container for shared transform resources."""

    llm_config: Optional[LLMConfig] = None
    keyword_extractor: Optional[KeywordExtractor] = None
    max_snippets: int = 5
    max_chars: int = 1800

    def ensure_keyword_extractor(self) -> KeywordExtractor:
        if self.keyword_extractor is None:
            if self.llm_config is None:
                raise RuntimeError(
                    "Keyword extractor requested but no LLMConfig was provided."
                )
            logger.debug("Creating keyword extractor for transform kernel.")
            self.keyword_extractor = KeywordExtractor(llm_config=self.llm_config)
        return self.keyword_extractor


def register_transform_ops(engine: ExecutionEngine, context: TransformContext) -> None:
    """
    Register transform kernels (query→query and code→query) on the execution engine.
    """

    engine.register(
        PhysicalOperator.QUERY_TRANSFORM.value, _query_to_query_kernel(context)
    )
    engine.register(
        PhysicalOperator.CODE_TO_QUERY_TRANSFORM.value, _code_to_query_kernel(context)
    )


def _query_to_query_kernel(context: TransformContext):
    def run(node: ExecutionNode, inputs: List[object]) -> Dict[str, object]:
        query = _resolve_query(node.params, inputs)
        extractor = context.ensure_keyword_extractor()
        logger.debug("Running query→query transform via keyword extractor.")

        result = extractor.extract_keywords(query)
        keywords = result.keywords
        joiner = node.params.get("joiner", " ")
        rewritten = joiner.join(keywords) if keywords else query

        return {
            "query": rewritten,
            "keywords": keywords,
            "source_query": query,
        }

    return run


def _code_to_query_kernel(context: TransformContext):
    def run(node: ExecutionNode, inputs: List[object]) -> Dict[str, object]:
        nodes = _collect_nodes(inputs)
        if not nodes:
            raise ValueError("Code-to-query transform received no code nodes.")

        max_snippets = int(node.params.get("max_snippets", context.max_snippets))
        max_chars = int(node.params.get("max_chars", context.max_chars))
        include_metadata = bool(node.params.get("include_metadata", True))
        joiner = node.params.get("joiner", "\n\n")

        snippets: List[str] = []
        sources: List[Dict[str, object]] = []

        for node_info in nodes[:max_snippets]:
            content = (node_info.content or "").strip()
            if not content:
                continue
            if len(content) > max_chars:
                content = content[: max_chars - 3] + "..."

            snippet = content
            if include_metadata:
                label_parts = [
                    node_info.file or node_info.node_name,
                    (
                        f"{(node_info.start_line or 0) + 1}"
                        if node_info.start_line is not None
                        else None
                    ),
                ]
                label = ":".join([part for part in label_parts if part])
                snippet = f"{label}\n{content}" if label else content

            snippets.append(snippet)
            sources.append(
                {
                    "file": node_info.file,
                    "node_name": node_info.node_name,
                    "start_line": node_info.start_line,
                    "end_line": node_info.end_line,
                    "score": node_info.score,
                }
            )

        if not snippets:
            raise ValueError("No usable snippets found for code-to-query transform.")

        rewritten = joiner.join(snippets)
        logger.debug(
            "Generated query from code snippets.",
            extra={"snippet_count": len(snippets)},
        )

        return {
            "query": rewritten,
            "sources": sources,
        }

    return run


def _resolve_query(params: Dict[str, object], inputs: List[object]) -> str:
    candidate_keys = ("query", "text", "prompt")
    for key in candidate_keys:
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for payload in inputs:
        query = _coerce_query_from_payload(payload)
        if query:
            return query

    raise ValueError("Query transform requires a source query string.")


def _coerce_query_from_payload(payload: object) -> Optional[str]:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("query", "text", "rewritten", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _collect_nodes(inputs: Sequence[object]) -> List[QueriedNode]:
    collected: List[QueriedNode] = []
    for payload in inputs:
        if isinstance(payload, QueriedNode):
            collected.append(payload)
            continue
        if isinstance(payload, dict):
            nodes = (
                payload.get("nodes")
                or payload.get("results")
                or payload.get("candidates")
            )
            if nodes:
                collected.extend(_collect_nodes(nodes))
            continue
        if isinstance(payload, (list, tuple)):
            collected.extend(_collect_nodes(list(payload)))
    return collected
