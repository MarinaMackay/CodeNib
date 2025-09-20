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
from llama_index.core.llms.llm import LLM
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.llms.vertex import Vertex

from ..log_utils import get_logger
from .utils import VertexAnthropicWithCredentials

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


@dataclass
class LLMConfig:
    """Configuration for a specific LLM model."""

    model_name: str
    provider: LLMProvider
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
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


def create_llm(config: LLMConfig, **kwargs) -> LLM:
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

    # Prepare common kwargs
    llm_kwargs = {
        "model": config.model_name,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        **kwargs,  # Allow override of any parameters
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
            base_url = config.get_config_value("OPENAI_API_BASE_URL")
            if base_url:
                llm_kwargs["api_base"] = base_url
            llm_class = OpenAI

        elif config.provider == LLMProvider.ANTHROPIC:
            llm_kwargs["api_key"] = config.get_config_value("ANTHROPIC_API_KEY")
            llm_class = Anthropic

        elif config.provider == LLMProvider.VERTEX_ANTHROPIC:
            credentials, project_id = config._get_vertex_credentials()
            # Get region, default to us-east5 if not specified
            region = os.environ.get("VERTEX_REGION", "us-east5")
            llm_kwargs.update(
                {
                    "credentials": credentials,
                    "project_id": project_id,
                    "region": region,
                }
            )
            llm_class = VertexAnthropicWithCredentials

        elif config.provider == LLMProvider.VERTEX_GEMINI:
            credentials, project_id = config._get_vertex_credentials()
            # Get location, default to us-central1 if not specified
            location = os.environ.get("VERTEX_REGION", "us-central1")
            llm_kwargs.update(
                {
                    "project": project_id,
                    "location": location,
                    "credentials": credentials,
                }
            )
            llm_class = Vertex

        else:
            raise LLMConfigurationError(f"Unsupported provider: {config.provider}")

        # Create and test the LLM
        llm = llm_class(**llm_kwargs)

        # Test the LLM with a simple completion
        try:
            _ = llm.complete("Say 'Hi'")
        except Exception as e:
            raise LLMConfigurationError(f"LLM test completion failed: {e}") from e

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


def get_llm(model: str, codeminer_config: Optional[LLMConfig] = None, **kwargs) -> LLM:
    """
    Legacy function for backward compatibility.

    Args:
        model: Model name (e.g., 'gpt-4', 'claude-3-opus-20240229')
        codeminer_config: LLMConfig instance. If None, creates a default one.
        **kwargs: Additional parameters to pass to the LLM constructor

    Returns:
        Configured LLM instance
    """
    if not model:
        raise LLMConfigurationError("Model name is required")

    if codeminer_config:
        config = codeminer_config
    else:
        # Try to infer provider from model name
        if "gpt" in model.lower() or "davinci" in model.lower():
            provider = LLMProvider.OPENAI
        elif "claude" in model.lower():
            provider = LLMProvider.ANTHROPIC
        elif "gemini" in model.lower():
            provider = LLMProvider.VERTEX_GEMINI
        else:
            provider = LLMProvider.OPENAI  # Default to OpenAI

        config = LLMConfig(model_name=model, provider=provider)

    return create_llm(config, **kwargs)
