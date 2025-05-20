from typing import Optional

from pydantic import BaseModel


class NodeAttributes(BaseModel):
    """Node attributes for graph nodes."""

    type: str = ""
    file: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
