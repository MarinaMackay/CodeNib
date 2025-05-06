"""
Keyword extraction agent for problem statements.
This module extracts key terms from problem statements using llama_index.
"""

import os
from pathlib import Path
from typing import List, Union

from llama_index.core.llms import ChatMessage
from pydantic import BaseModel, Field

from .gen_config import Config, get_llm
from .log_utils import get_logger

logger = get_logger(__name__)


# Define Pydantic model for structured output
class KeywordExtraction(BaseModel):
    """Model for keyword extraction output."""

    keywords: List[str] = Field(description="List of extracted keywords")


class KeywordExtractor:
    """Agent for extracting keywords from problem statements."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        config_path: Union[Path, str] = "key.cfg",
        temperature: float = 0.0,
        **kwargs,
    ):
        """
        Initialize the keyword extractor.

        Args:
            model_name (str): Name of the LLM model to use
            config_path (Union[Path, str]): Path to the configuration file
            temperature (float): Temperature for the LLM, lower is more deterministic
            **kwargs: Additional arguments to pass to the LLM
        """
        # Convert Path object to string if needed, but preserve relative paths
        if isinstance(config_path, Path):
            config_path = str(config_path)
            
        # Load configuration
        codeminer_config = Config(config_path)

        # Get the LLM using the gen_config helper
        self.llm = get_llm(
            model=model_name,
            codeminer_config=codeminer_config,
            temperature=temperature,
            **kwargs,
        )

        # Convert the LLM to a structured output LLM
        self.structured_llm = self.llm.as_structured_llm(output_cls=KeywordExtraction)

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
            "class names, modules, and concepts that would be useful for searching in a codebase. "
            "\n\n"
            "Guidelines for extraction:\n"
            "1. Identify technical terms and code elements (functions, classes, modules)\n"
            "2. Include both specific terms and broader concepts\n"
            "3. Prefer precise terms over general ones\n"
            "4. Remove common stopwords and general programming terms\n"
            "5. Consider variations of terms when appropriate\n"
            "6. Return a list of keywords separated by commas\n"
            "\n\n"
            "Please extract the key technical terms and concepts from "
            "the following problem statement:\n\n"
            f"{problem_statement}\n\n"
            "Return only the essential terms that would be most useful for searching in a codebase."
        )

        # Use structured LLM to get output directly as a KeywordExtraction object
        input_msg = ChatMessage.from_str(role="user", content=prompt)
        result = self.structured_llm.chat([input_msg])
        logger.debug(f"Extracted keywords: {result.raw}")
        return result.raw


def extract_keywords_from_statement(
    problem_statement: str,
    model_name: str = "gpt-4o",
    config_path: Union[Path, str] = "key.cfg",
    temperature: float = 0.0,
    **kwargs,
) -> KeywordExtraction:
    """
    Extract keywords from a problem statement.

    Args:
        problem_statement (str): The problem statement to extract keywords from
        model_name (str): Name of the LLM model to use
        config_path (Union[Path, str]): Path to the configuration file
        temperature (float): Temperature for the LLM, lower is more deterministic
        **kwargs: Additional arguments to pass to the LLM

    Returns:
        KeywordExtraction: Structured output with extracted keywords
    """
    extractor = KeywordExtractor(
        model_name=model_name,
        config_path=config_path,
        temperature=temperature,
        **kwargs,
    )
    return extractor.extract_keywords(problem_statement)
