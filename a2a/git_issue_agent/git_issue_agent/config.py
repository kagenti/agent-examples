import json
import os
from pydantic_settings import BaseSettings
from pydantic import model_validator
from pydantic import Field
from typing import Literal, Optional


class Settings(BaseSettings):
    LOG_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        os.getenv("LOG_LEVEL", "INFO"),
        description="Application log level",       
    )
    TASK_MODEL_ID: str = Field(
        os.getenv("TASK_MODEL_ID", "granite3.3:8b"),
        description="The ID of the task model",
    )
    LLM_API_BASE: str = Field(
        os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
        description="The URL for OpenAI API",
    )
    LLM_API_KEY: str = Field(os.getenv("LLM_API_KEY", "my_api_key"), description="The key for OpenAI API")
    EXTRA_HEADERS: dict = Field({}, description="Extra headers for the OpenAI API")
    MODEL_TEMPERATURE: float = Field(
        os.getenv("MODEL_TEMPERATURE", 0),
        description="The temperature for the model",
        ge=0,
    )
    MCP_URL: str = Field(os.getenv("MCP_URL", "https://api.githubcopilot.com/mcp/"), description="Endpoint for an option MCP server")
    SERVICE_PORT: int = Field(os.getenv("SERVICE_URL", 8000), description="Port on which the service will run.")
    GITHUB_TOKEN: str = Field(os.getenv("GITHUB_TOKEN", None), description="If not using agent with authorization, the default Github token to use")

    # auth variables for token validation
    ISSUER: Optional[str] = Field(
        os.getenv("ISSUER", None),
        description="The issuer for incoming JWT tokens"
    )
    JWKS_URI: Optional[str] = Field(
        os.getenv("JWKS_URI", None),
        description="Endpoint to obtain JWKS from auth server"
    )
    AUDIENCE: Optional[str] = Field(
        os.getenv("AUDIENCE", None),
        description="Expected audience value during resource validation"
    )

    # auth variables for token exchange
    TOKEN_URL: Optional[str] = Field(
        os.getenv("TOKEN_URL", None),
        description="Token endpoint to obtain new access tokens"
    )
    CLIENT_ID: Optional[str] = Field(
        os.getenv("CLIENT_ID", None),
        description="Client ID to authenticate to OAuth server"
    )
    CLIENT_SECRET: Optional[str] = Field(
        os.getenv("CLIENT_SECRET", None),
        description="Client secret to authenticate to OAuth server"
    )
    TARGET_AUDIENCE: Optional[str] = Field(
        os.getenv("TARGET_AUDIENCE", None),
        description="Target audience to request during token exchange"
    )
    TARGET_SCOPES: Optional[str] = Field(
        os.getenv("TARGET_SCOPES", None),
        description="Target scopes to request during token exchange"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @model_validator(mode="after")
    def validate_extra_headers(self) -> "Settings":
        if os.getenv("EXTRA_HEADERS"):
            try:
                self.EXTRA_HEADERS = json.loads(os.getenv("EXTRA_HEADERS"))
            except json.JSONDecodeError:
                raise ValueError("EXTRA_HEADERS must be a valid JSON string")

        return self

settings = Settings()  # type: ignore[call-arg]
