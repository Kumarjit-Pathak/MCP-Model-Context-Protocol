"""
Client Configuration - Manages settings and environment variables
"""

from pydantic_settings import BaseSettings
from typing import Literal


class ClientSettings(BaseSettings):
    """Client configuration loaded from environment variables."""

    # LLM Provider Selection
    llm_provider: Literal["anthropic", "openai"] = "openai"

    # Anthropic Configuration
    anthropic_api_key: str = ""

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "openai/gpt-4o-mini"

    # Client Settings
    debug: bool = True
    max_iterations: int = 10
    timeout: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env


def get_client_settings() -> ClientSettings:
    """Get client settings instance."""
    return ClientSettings()
