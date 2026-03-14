"""
LLM Configuration and Factory Module for CodeMiner.

This module provides a centralized configuration system for managing different LLM
providers via LiteLLM.  It supports OpenAI, Anthropic, Google Vertex AI (Gemini and
Anthropic), and vLLM (OpenAI-compatible servers).
"""

import os
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from ..log_utils import get_logger
from .litellm_chat import LiteLLMChat, human_message

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
            logger.debug(f"Found {key!r} in config data: {self.config_data.get(key)}")
            return self.config_data.get(key)
        if key in os.environ:
            return os.environ[key]

        raise LLMConfigurationError(
            f"Configuration key {key!r} not found in config data or environment variables"
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
            logger.debug(f"Found {key!r} in config data: {self.config_data.get(key)}")
            return self.config_data.get(key)
        if key in os.environ:
            return os.environ[key]
        return None

    def _get_vertex_credentials(self) -> tuple:
        """Get Google Cloud credentials and project ID.

        Supports both service account JSON files (via
        GOOGLE_APPLICATION_CREDENTIALS) and Application Default Credentials
        (via ``gcloud auth application-default login``).
        """
        try:
            sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if sa_path:
                sa_path = os.path.expanduser(sa_path)
                if not os.path.exists(sa_path):
                    raise LLMConfigurationError(
                        f"Google Cloud credentials file not found: {sa_path}"
                    )
                # Try service account first
                try:
                    credentials = service_account.Credentials.from_service_account_file(
                        sa_path,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                    return credentials, credentials.project_id
                except ValueError:
                    pass  # Not a service account file, fall through to ADC

            # Fall back to Application Default Credentials
            import google.auth
            credentials, project_id = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            if not project_id:
                project_id = getattr(credentials, "quota_project_id", None)
            if not project_id:
                project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
            if not project_id:
                raise LLMConfigurationError(
                    "Could not determine Google Cloud project ID. "
                    "Run 'gcloud config set project <PROJECT_ID>' or set GOOGLE_CLOUD_PROJECT."
                )
            return credentials, project_id

        except Exception as e:
            if isinstance(e, LLMConfigurationError):
                raise
            raise LLMConfigurationError(
                f"Failed to load Google Cloud credentials: {e}. "
                "Either set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON, "
                "or run 'gcloud auth application-default login'."
            ) from e


def create_llm(config: LLMConfig, **kwargs) -> LiteLLMChat:
    """
    Factory function to create LLM instances backed by LiteLLM.

    Args:
        config: LLM configuration instance.
        **kwargs: Additional parameters forwarded to LiteLLMChat.

    Returns:
        Configured LiteLLMChat instance

    Raises:
        LLMConfigurationError: If there's an error in configuration or initialization
    """
    if config is None:
        raise LLMConfigurationError("LLMConfig instance is required")

    # Filter out codeminer-specific kwargs that shouldn't be passed to the LLM
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != "codeminer_config"}

    api_key: Optional[str] = None
    api_base: Optional[str] = None
    extra_kwargs: Dict[str, Any] = {}

    if config.timeout:
        extra_kwargs["timeout"] = config.timeout
    if config.max_retries:
        extra_kwargs["num_retries"] = config.max_retries

    try:
        if config.provider == LLMProvider.OPENAI:
            model = config.model_name
            api_key = config.get_config_value("OPENAI_API_KEY")
            api_base = config.get_config_value_optional("OPENAI_API_BASE_URL")

        elif config.provider == LLMProvider.ANTHROPIC:
            model = config.model_name
            api_key = config.get_config_value("ANTHROPIC_API_KEY")

        elif config.provider == LLMProvider.VERTEX_ANTHROPIC:
            model = f"vertex_ai/{config.model_name}"
            project_id = config._get_vertex_project()
            region = os.environ.get("VERTEX_REGION", "us-east5")
            extra_kwargs["vertex_project"] = project_id
            extra_kwargs["vertex_location"] = region

        elif config.provider == LLMProvider.VERTEX_GEMINI:
            model = f"vertex_ai/{config.model_name}"
            project_id = config._get_vertex_project()
            location = os.environ.get("VERTEX_REGION", "us-central1")
            extra_kwargs["vertex_project"] = project_id
            extra_kwargs["vertex_location"] = location
            if config.thinking_budget:
                extra_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": config.thinking_budget,
                }

        elif config.provider == LLMProvider.VLLM:
            # Local vLLM is no longer supported — fall through to VLLM_OPENAI
            warnings.warn(
                "LLMProvider.VLLM (local in-process) is deprecated. "
                "Use VLLM_OPENAI with a vLLM OpenAI-compatible server instead. "
                "Falling back to VLLM_OPENAI behavior.",
                DeprecationWarning,
                stacklevel=2,
            )
            model = f"openai/{config.model_name}"
            api_base = config.get_config_value_optional("VLLM_API_BASE_URL")
            if not api_base:
                api_base = "http://localhost:9000/v1"
            api_key = config.get_config_value_optional("VLLM_API_KEY")
            if not api_key:
                api_key = "token-abc123"

        elif config.provider == LLMProvider.VLLM_OPENAI:
            model = f"openai/{config.model_name}"
            api_base = config.get_config_value_optional("VLLM_API_BASE_URL")
            if not api_base:
                api_base = "http://localhost:9000/v1"
            api_key = config.get_config_value_optional("VLLM_API_KEY")
            if not api_key:
                api_key = "token-abc123"

        else:
            raise LLMConfigurationError(f"Unsupported provider: {config.provider}")

        # Merge caller-supplied overrides
        extra_kwargs.update(filtered_kwargs)

        llm = LiteLLMChat(
            model=model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=api_key,
            api_base=api_base,
            extra_kwargs=extra_kwargs,
        )

        # Test the LLM with a simple invocation
        try:
            _ = llm.invoke([human_message("Say 'Hi'")])
        except Exception as e:
            raise LLMConfigurationError(f"LLM test invocation failed: {e}") from e

        return llm

    except Exception as e:
        if isinstance(e, LLMConfigurationError):
            raise
        raise LLMConfigurationError(
            f"Failed to create LLM {config.model_name!r}: {e}"
        ) from e


# Legacy compatibility
class Config(LLMConfig):
    """Legacy Config class for backward compatibility."""

    def __getitem__(self, key: str) -> str:
        """Legacy method for accessing config values."""
        return self.get_config_value(key)
