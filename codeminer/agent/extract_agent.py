"""
Keyword extraction agent for problem statements.
This module extracts key terms from problem statements using llama_index.
"""

from typing import List

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ..llm.llm_config import LLMConfig, create_llm
from ..log_utils import get_logger

logger = get_logger(__name__)


# Define Pydantic model for structured output
class KeywordExtraction(BaseModel):
    """Model for keyword extraction output."""

    keywords: List[str] = Field(description="List of extracted keywords")


class KeywordExtractor:
    """Agent for extracting keywords from problem statements."""

    def __init__(
        self,
        llm_config: LLMConfig,
        **kwargs,
    ):
        """Initialize the keyword extractor.

        Args:
            llm_config: LLM configuration containing model, provider, and other settings.
            **kwargs: Additional keyword arguments forwarded to the LLM factory.
        """
        self.llm_config = llm_config

        # Build the LLM instance using the configuration
        self.llm = create_llm(
            config=llm_config,
            **kwargs,
        )

        # Convert the LLM to a structured output LLM using LangChain style
        self.structured_llm = self.llm.with_structured_output(KeywordExtraction)

    def extract_keywords(self, problem_statement: str) -> KeywordExtraction:
        """
        Extract keywords from a problem statement.

        Args:
            problem_statement (str): The problem statement to extract keywords from

        Returns:
            KeywordExtraction: Structured output with extracted keywords
        """
        # Create prompt with detailed instructions
        prompt = (
            "You are a keyword extraction specialist. Your task is to extract important keywords "
            "from problem statements. Focus on identifying technical terms, function names, "
            "class names, modules, file paths, and concepts that would be useful for searching in a codebase. "
            "\n\n"
            "Guidelines for extraction:\n"
            "1. Extract file paths and file names (e.g., 'django/db/models/expressions.py'-> and 'expressions.py')\n"
            "2. Extract function and method names (e.g., 'separability_matrix', 'run_validators')\n"
            "3. Extract class names and module names\n"
            "4. Prefer precise terms over general ones\n"
            "5. Remove common stopwords and general programming terms\n"
            "\n\n"
            "Please extract the key technical terms and concepts from "
            "the following problem statement:\n\n"
            f"{problem_statement}\n\n"
            "Return only the essential terms that would be most useful for searching in a codebase."
        )

        # Use structured LLM to get output directly as a KeywordExtraction object
        input_msg = HumanMessage(content=prompt)
        result = self.structured_llm.invoke([input_msg])
        logger.debug(f"Extracted keywords: {result}")
        return result


def extract_keywords_from_statement(
    problem_statement: str,
    llm_config: LLMConfig,
    **kwargs,
) -> KeywordExtraction:
    """
    Extract keywords from a problem statement.

    Args:
        problem_statement (str): The problem statement to extract keywords from
        llm_config (LLMConfig): LLM configuration containing model, provider, and other settings.
        **kwargs: Additional arguments forwarded to the LLM factory

    Returns:
        KeywordExtraction: Structured output with extracted keywords
    """
    extractor = KeywordExtractor(
        llm_config=llm_config,
        **kwargs,
    )
    return extractor.extract_keywords(problem_statement)
