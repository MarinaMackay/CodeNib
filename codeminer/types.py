from typing import Optional

from pydantic import BaseModel


class NodeInfo(BaseModel):
    """Node attributes for graph nodes."""

    node_name: str = ""
    type: str = ""
    file: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None


class NodeWithContent(NodeInfo):
    """Node attributes with content."""

    content: str = ""


class NodeWithScore(NodeInfo):
    """Node attributes with relevance score."""

    score: float = 0.0
