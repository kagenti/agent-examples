"""Configuration settings for Simple Generalist Agent."""

import json
import os
from typing import Any, Literal, Optional
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Logging
    LOG_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        default=os.getenv("LOG_LEVEL", "INFO"),  # type: ignore[arg-type]
        description="Application log level",
    )
    
    # A2A Server Configuration
    A2A_HOST: str = Field(
        default=os.getenv("A2A_HOST", "0.0.0.0"),
        description="Host address for A2A server",
    )
    A2A_PORT: int = Field(
        default=int(os.getenv("A2A_PORT", "8000")),
        description="Port for A2A server",
    )
    
    # MCP Server Configuration
    MCP_SERVER_URL: str = Field(
        default=os.getenv("MCP_SERVER_URL", ""),
        description="MCP server URL",
        validation_alias=AliasChoices("MCP_SERVER_URL", "MCP_SERVERS"),
    )
    
    # LLM Configuration
    LLM_MODEL: str = Field(
        default=os.getenv("LLM_MODEL", "gpt-4"),
        description="LLM model name",
    )
    LLM_API_KEY: Optional[str] = Field(
        default=os.getenv("LLM_API_KEY"),
        description="API key for LLM provider",
    )
    LLM_BASE_URL: Optional[str] = Field(
        default=os.getenv("LLM_BASE_URL"),
        description="Base URL for LLM API (for custom endpoints)",
    )
    LLM_TEMPERATURE: float = Field(
        default=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        description="Temperature for LLM sampling",
    )
    # Execution Limits
    MAX_ITERATIONS: int = Field(
        default=int(os.getenv("MAX_ITERATIONS", "20")),
        description="Maximum number of agent loop iterations",
    )
    A2A_PUBLIC_URL: Optional[str] = Field(
        default=os.getenv("A2A_PUBLIC_URL"),
        description="Publicly routable A2A base URL for agent discovery",
    )
    EXTRA_HEADERS: dict[str, str] = Field(
        default_factory=dict,
        description="Extra headers for the OpenAI API (JSON string, e.g. '{\"key\": \"value\"}')",
    )

    @field_validator("EXTRA_HEADERS", mode="before")
    @classmethod
    def _parse_extra_headers(cls, v: Any) -> dict[str, str]:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return {}
            return json.loads(v)
        if v is None:
            return {}
        return v

    OTEL_CONSOLE_TRACING: bool = Field(
        default=os.getenv("OTEL_CONSOLE_TRACING", "false").lower() in ("true", "1", "yes"),
        description="Print OpenTelemetry traces to console when no OTLP endpoint is configured",
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def load_settings() -> Settings:
    """Load and return application settings."""
    return Settings()  # type: ignore[call-arg]

# Made with Bob
