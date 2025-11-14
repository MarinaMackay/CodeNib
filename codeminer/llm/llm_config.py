"""
LLM Configuration and Factory Module for CodeMiner.

This module provides a centralized configuration system for managing different LLM providers
including OpenAI, Anthropic, and Google Vertex AI. It supports multiple authentication methods
and provider-specific configurations.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from google.oauth2 import service_account
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import VLLM
from langchain_core.language_models import BaseLLM
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_openai import ChatOpenAI

from ..log_utils import get_logger

logger = get_logger(__name__)


class LLMConfigurationError(Exception):
    """Exception raised when there's an error in LLM configuration."""

    pass


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VERTEX_ANTHROPIC = "vertexanthropic"
    VERTEX_GEMINI = "vertexgemini"
    VLLM = "vllm"
    VLLM_OPENAI = "vllm_openai"  # vLLM with OpenAI-compatible API


@dataclass
class LLMConfig:
    """Configuration for a specific LLM model."""

    model_name: str
    provider: LLMProvider
    max_tokens: Optional[int] = 8192
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = 0.0
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    thinking_budget: Optional[int] = None
    config_data: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize configuration data if not provided."""
        if self.config_data is None:
            self.config_data = {}

    def get_config_value(self, key: str) -> str:
        """
        Get configuration value with priority: config_data > env vars.

        Args:
            key: Configuration key to retrieve

        Returns:
            Configuration value

        Raises:
            LLMConfigurationError: If key is not found in any source
        """
        # Priority: config_data > environment variables
        if self.config_data.get(key):
            logger.debug(f"Found '{key}' in config data: {self.config_data.get(key)}")
            return self.config_data.get(key)
        if key in os.environ:
            return os.environ[key]

        raise LLMConfigurationError(
            f"Configuration key '{key}' not found in config data or environment variables"
        )

    def get_config_value_optional(self, key: str) -> Optional[str]:
        """
        Get configuration value with priority: config_data > env vars.
        Returns None if not found instead of raising an exception.

        Args:
            key: Configuration key to retrieve

        Returns:
            Configuration value or None if not found
        """
        # Priority: config_data > environment variables
        if self.config_data.get(key):
            logger.debug(f"Found '{key}' in config data: {self.config_data.get(key)}")
            return self.config_data.get(key)
        if key in os.environ:
            return os.environ[key]
        return None

    def _get_vertex_credentials(self) -> tuple:
        """Get Google Cloud service account credentials and project ID."""
        try:
            # Get credentials from GOOGLE_APPLICATION_CREDENTIALS env var
            if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
                raise LLMConfigurationError(
                    "GOOGLE_APPLICATION_CREDENTIALS environment variable is required for Vertex AI. "
                    "Set it to point to your service account JSON file."
                )

            service_account_path = os.path.expanduser(
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
            )
            if not os.path.exists(service_account_path):
                raise LLMConfigurationError(
                    f"Google Cloud Service Account file not found: {service_account_path}"
                )

            credentials = service_account.Credentials.from_service_account_file(
                service_account_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            return credentials, credentials.project_id

        except Exception as e:
            if isinstance(e, LLMConfigurationError):
                raise
            raise LLMConfigurationError(
                f"Failed to load service account credentials: {e}"
            ) from e


def create_llm(config: LLMConfig, **kwargs) -> BaseLLM:
    """
    Factory function to create LLM instances.

    Args:
        model: Model name (e.g., 'gpt-4', 'claude-3-opus-20240229', 'gemini-1.5-pro')
        config: LLM configuration instance. If None, creates a default one.
        **kwargs: Additional parameters to pass to the LLM constructor

    Returns:
        Configured LLM instance

    Raises:
        LLMConfigurationError: If there's an error in configuration or initialization
    """
    if config is None:
        # error if config is None
        raise LLMConfigurationError("LLMConfig instance is required")

    # Filter out codeminer-specific kwargs that shouldn't be passed to LLM constructors
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != "codeminer_config"}

    # Prepare common kwargs
    llm_kwargs = {
        "model": config.model_name,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        **filtered_kwargs,  # Allow override of any parameters
    }

    # Add model-specific defaults if not provided
    if config.max_tokens and "max_tokens" not in llm_kwargs:
        llm_kwargs["max_tokens"] = config.max_tokens
    if config.temperature and "temperature" not in llm_kwargs:
        llm_kwargs["temperature"] = config.temperature
    if config.timeout and "timeout" not in llm_kwargs:
        llm_kwargs["timeout"] = config.timeout
    if config.max_retries and "max_retries" not in llm_kwargs:
        llm_kwargs["max_retries"] = config.max_retries

    try:
        if config.provider == LLMProvider.OPENAI:
            llm_kwargs["api_key"] = config.get_config_value("OPENAI_API_KEY")
            base_url = config.get_config_value_optional("OPENAI_API_BASE_URL")
            if base_url:
                llm_kwargs["base_url"] = base_url
            llm_class = ChatOpenAI

        elif config.provider == LLMProvider.ANTHROPIC:
            llm_kwargs["api_key"] = config.get_config_value("ANTHROPIC_API_KEY")
            llm_class = ChatAnthropic

        elif config.provider == LLMProvider.VERTEX_ANTHROPIC:
            if ChatAnthropicVertex is None:
                raise LLMConfigurationError(
                    "ChatAnthropicVertex is not available. Please install "
                    "the latest langchain-google-vertexai package."
                )
            credentials, project_id = config._get_vertex_credentials()
            region = os.environ.get("VERTEX_REGION", "us-east5")
            llm_kwargs.update(
                {
                    "project": project_id,
                    "location": region,
                    "credentials": credentials,
                }
            )
            # Use Claude models on Vertex AI
            llm_class = ChatAnthropicVertex

        elif config.provider == LLMProvider.VERTEX_GEMINI:
            credentials, project_id = config._get_vertex_credentials()
            location = os.environ.get("VERTEX_REGION", "us-central1")
            llm_kwargs.update(
                {
                    "project": project_id,
                    "location": location,
                    "credentials": credentials,
                }
            )
            # Add thinking budget for Gemini models > 2.5
            if config.thinking_budget:
                llm_kwargs["thinking_budget"] = config.thinking_budget
            llm_class = ChatVertexAI

        elif config.provider == LLMProvider.VLLM:
            # VLLM specific configuration
            vllm_kwargs = {
                "model": config.model_name,
                "temperature": (
                    config.temperature if config.temperature is not None else 0.8
                ),
            }

            # Add VLLM-specific parameters from config or use defaults
            trust_remote_code = config.get_config_value_optional(
                "VLLM_TRUST_REMOTE_CODE"
            )
            if trust_remote_code is not None:
                vllm_kwargs["trust_remote_code"] = trust_remote_code.lower() == "true"
            else:
                vllm_kwargs["trust_remote_code"] = True  # Default for HF models

            # Map max_tokens to max_new_tokens for VLLM
            if config.max_tokens:
                vllm_kwargs["max_new_tokens"] = config.max_tokens

            # VLLM-specific sampling parameters
            top_k = config.get_config_value_optional("VLLM_TOP_K")
            if top_k:
                vllm_kwargs["top_k"] = int(top_k)

            top_p = config.get_config_value_optional("VLLM_TOP_P")
            if top_p:
                vllm_kwargs["top_p"] = float(top_p)
            # logger.debug(f"VLLM top_p: {top_p}")
            llm_kwargs = vllm_kwargs
            llm_class = VLLM

        elif config.provider == LLMProvider.VLLM_OPENAI:
            # vLLM with OpenAI-compatible API server
            # Use OpenAI client but with vLLM server endpoint
            base_url = config.get_config_value_optional("VLLM_API_BASE_URL")
            if not base_url:
                base_url = "http://localhost:9000/v1"  # Default vLLM server endpoint

            api_key = config.get_config_value_optional("VLLM_API_KEY")
            if not api_key:
                api_key = "token-abc123"  # Default dummy key for vLLM server

            llm_kwargs.update(
                {
                    "api_key": api_key,
                    "base_url": base_url,
                }
            )
            llm_class = ChatOpenAI

        else:
            raise LLMConfigurationError(f"Unsupported provider: {config.provider}")

        # Create and test the LLM
        llm = llm_class(**llm_kwargs)

        # Test the LLM with a simple invocation
        try:
            from langchain_core.messages import HumanMessage

            _ = llm.invoke([HumanMessage(content="Say 'Hi'")])
        except Exception as e:
            raise LLMConfigurationError(f"LLM test invocation failed: {e}") from e

        return llm

    except Exception as e:
        if isinstance(e, LLMConfigurationError):
            raise
        raise LLMConfigurationError(
            f"Failed to create LLM '{config.model_name}': {e}"
        ) from e


# Legacy compatibility
class Config(LLMConfig):
    """Legacy Config class for backward compatibility."""

    def __getitem__(self, key: str) -> str:
        """Legacy method for accessing config values."""
        return self.get_config_value(key)
