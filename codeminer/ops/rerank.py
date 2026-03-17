from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..agent.rerank_agent import RerankAgent
from ..index.embedding.vector_store import CodeVectorStore
from ..llm.litellm_chat import LiteLLMChat
from ..log_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RerankContext:
    """Shared context carrying the rerank agent and configuration."""

    llm: Optional[LiteLLMChat] = None
    agent: Optional[RerankAgent] = None
    embedding_store: Optional[CodeVectorStore] = None
    top_k: Optional[int] = None
    candidate_top_k: Optional[int] = None
    window_size: Optional[int] = None
    window_step: Optional[int] = None

    def ensure_agent(self) -> RerankAgent:
        if self.agent is None:
            if self.llm is None:
                raise RuntimeError("Rerank agent requested but no LLM was provided.")
            logger.info(
                "Creating rerank agent.",
                extra={"model": self.llm.model},
            )
            self.agent = RerankAgent(llm=self.llm)
        return self.agent
