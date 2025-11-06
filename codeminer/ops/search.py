from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..index.embedding.vector_store import CodeVectorStore
from ..index.regex_idx.regex_idx import RegexNodeIndex
from ..index.sparse_idx.bm25_index import BM25CodeIndexer
from ..log_utils import get_logger
from ..plans.execution import ExecutionEngine
from ..plans.ir_exec import ExecutionNode
from ..plans.ir_physical import PhysicalOperator
from ..types import NodeWithContent, NodeWithScore, NodeWithScoreContent

logger = get_logger(__name__)


@dataclass
class SearchContext:
    """Handles backing indexes for registered search operators."""

    bm25: Optional[BM25CodeIndexer] = None
    vector_store: Optional[CodeVectorStore] = None
    regex_index: Optional[RegexNodeIndex] = None
    default_top_k: int = 10


def register_search_ops(engine: ExecutionEngine, context: SearchContext) -> None:
    """
    Register search-related execution kernels on the provided engine.

    The function wires physical operator names to concrete kernels so that the
    lowering pipeline can execute BM25, vector, regex, and hybrid searches.
    """

    engine.register(PhysicalOperator.BM25_TOPK.value, _bm25_kernel(context))
    engine.register(PhysicalOperator.FAISS_SEARCH.value, _vector_kernel(context))
    engine.register(PhysicalOperator.HYBRID_SEARCH.value, _hybrid_kernel())
    engine.register(PhysicalOperator.REGEX_SEARCH.value, _regex_kernel(context))


def _bm25_kernel(context: SearchContext):
    def run(node: ExecutionNode, _: List[object]) -> List[NodeWithScoreContent]:
        index = context.bm25
        if index is None:
            raise RuntimeError("BM25 operator invoked but no BM25CodeIndexer provided.")

        query = _extract_query(node.params)
        top_k = _resolve_top_k(node.params, context.default_top_k)
        include_content = bool(node.params.get("return_content", False))
        filter_test = bool(node.params.get("filter_test", False))
        wrap_with_ln = bool(node.params.get("wrap_with_line_numbers", True))

        logger.debug(
            "Executing BM25 search",
            extra={"query": query, "k": top_k, "filter_test": filter_test},
        )

        raw_results = index.search(
            query=query,
            top_k=top_k,
            return_code_content=include_content,
            wrap_with_ln=wrap_with_ln,
            filter_test=filter_test,
        )
        return _to_score_content(raw_results)

    return run


def _vector_kernel(context: SearchContext):
    def run(node: ExecutionNode, _: List[object]) -> List[NodeWithScoreContent]:
        store = context.vector_store
        if store is None:
            raise RuntimeError(
                "Vector search operator invoked without a CodeVectorStore."
            )

        query = _extract_query(node.params)
        top_k = _resolve_top_k(node.params, context.default_top_k)
        score_threshold = node.params.get("score_threshold")
        include_content = bool(node.params.get("return_content", False))

        logger.debug(
            "Executing vector search",
            extra={"query": query, "k": top_k, "score_threshold": score_threshold},
        )

        scored = store.search(query=query, top_k=top_k, score_threshold=score_threshold)

        content_map: Dict[Tuple[str, str, Optional[int], Optional[int]], str] = {}
        if include_content:
            with_content = store.search_with_content(
                query=query, top_k=top_k, score_threshold=score_threshold
            )
            for item in with_content:
                data = _dump_model(item)
                key = (
                    data.get("file"),
                    data.get("node_name"),
                    data.get("start_line"),
                    data.get("end_line"),
                )
                content = data.get("content")
                if content is not None:
                    content_map[key] = content

        normalized: List[NodeWithScoreContent] = []
        for result in scored:
            data = _dump_model(result)
            key = (
                data.get("file"),
                data.get("node_name"),
                data.get("start_line"),
                data.get("end_line"),
            )
            data["content"] = content_map.get(key)
            normalized.append(NodeWithScoreContent(**data))

        return normalized

    return run


def _regex_kernel(context: SearchContext):
    def run(node: ExecutionNode, _: List[object]) -> List[NodeWithScoreContent]:
        index = context.regex_index
        if index is None:
            raise RuntimeError("Regex search operator invoked without RegexNodeIndex.")

        pattern = node.params.get("pattern") or _extract_query(node.params)
        top_k = _resolve_top_k(node.params, context.default_top_k)
        file_glob = node.params.get("file_glob")
        node_type = node.params.get("node_type")
        case_sensitive = bool(node.params.get("case_sensitive", False))
        use_regex = bool(node.params.get("use_regex", True))

        logger.debug(
            "Executing regex search",
            extra={
                "pattern": pattern,
                "file_glob": file_glob,
                "node_type": node_type,
                "case_sensitive": case_sensitive,
                "use_regex": use_regex,
            },
        )

        raw_results = index.search(
            pattern=pattern,
            file_glob=file_glob,
            node_type=node_type,
            case_sensitive=case_sensitive,
            use_regex=use_regex,
        )

        converted = _to_score_content(raw_results)
        if top_k:
            converted = converted[:top_k]
        return converted

    return run


def _hybrid_kernel():
    def run(node: ExecutionNode, inputs: List[object]) -> List[NodeWithScoreContent]:
        if not inputs:
            return []

        branches: List[List[NodeWithScoreContent]] = []
        for branch in inputs:
            branches.append(_to_score_content(_ensure_sequence(branch)))

        weights = node.params.get("weights") or []
        if not weights or len(weights) != len(branches):
            # Default to uniform weights when unspecified or mismatched.
            weights = [1.0] * len(branches)

        accumulator: Dict[
            Tuple[str, str, Optional[int], Optional[int]], NodeWithScoreContent
        ] = {}

        for weight, results in zip(weights, branches):
            for rank, item in enumerate(results):
                base_score = item.score or 0.0
                if base_score == 0.0:
                    base_score = 1.0 / (rank + 1)

                key = (
                    item.file,
                    item.node_name,
                    item.start_line,
                    item.end_line,
                )
                weighted = weight * base_score

                if key not in accumulator:
                    accumulator[key] = _with_score(item, weighted)
                    continue

                existing = accumulator[key]
                accumulator[key] = _merge_scores(existing, item, weighted)

        merged = sorted(
            accumulator.values(),
            key=lambda node: node.score,
            reverse=True,
        )

        top_k = _resolve_top_k(node.params, len(merged))
        return merged[:top_k]

    return run


def _extract_query(params: Dict[str, object]) -> str:
    for key in ("query", "pattern", "text"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("Search operator requires a non-empty 'query' string parameter.")


def _resolve_top_k(params: Dict[str, object], default_k: int) -> int:
    for key in ("top_k", "k", "limit"):
        value = params.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return default_k


def _ensure_sequence(value: object) -> Sequence[object]:
    if isinstance(value, (list, tuple)):
        return value
    return [value]


def _to_score_content(results: Sequence[object]) -> List[NodeWithScoreContent]:
    converted: List[NodeWithScoreContent] = []
    for rank, item in enumerate(results):
        if isinstance(item, NodeWithScoreContent):
            converted.append(item)
            continue
        if isinstance(item, NodeWithScore):
            data = _dump_model(item)
            data.setdefault("content", None)
            converted.append(NodeWithScoreContent(**data))
            continue
        if isinstance(item, NodeWithContent):
            data = _dump_model(item)
            score = data.get("score") or 0.0
            if not score:
                score = 1.0 / (rank + 1)
            data["score"] = score
            converted.append(NodeWithScoreContent(**data))
            continue
        if isinstance(item, dict):
            data = dict(item)
            score = data.get("score") or 0.0
            if not score:
                score = 1.0 / (rank + 1)
            data["score"] = score
            data.setdefault("node_name", data.get("name", ""))
            data.setdefault("content", data.get("content"))
            converted.append(NodeWithScoreContent(**data))
            continue
        raise TypeError(f"Unsupported result type for normalization: {type(item)}")
    return converted


def _merge_scores(
    existing: NodeWithScoreContent, new_item: NodeWithScoreContent, delta_score: float
) -> NodeWithScoreContent:
    updated_score = existing.score + delta_score
    content = existing.content or new_item.content
    return _with_score(existing, updated_score, content_override=content)


def _with_score(
    item: NodeWithScoreContent, score: float, *, content_override: Optional[str] = None
) -> NodeWithScoreContent:
    update = {"score": score}
    if content_override is not None:
        update["content"] = content_override
    if hasattr(item, "model_copy"):
        return item.model_copy(update=update)
    if hasattr(item, "copy"):
        return item.copy(update=update)
    data = _dump_model(item)
    data.update(update)
    return NodeWithScoreContent(**data)


def _dump_model(model: object) -> Dict[str, object]:
    if hasattr(model, "model_dump"):
        return dict(model.model_dump())
    if hasattr(model, "dict"):
        return dict(model.dict())
    raise TypeError(f"Object {model} does not support model dumping.")
