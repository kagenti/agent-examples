import json
import logging
import os
import sys
from pydantic_settings import BaseSettings
from pydantic import model_validator
from pydantic import Field
from typing import Literal, Optional


class Settings(BaseSettings):
    LOG_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        os.getenv("LOG_LEVEL", "DEBUG"),
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
    MAX_PLAN_STEPS: int = Field(
        os.getenv("MAX_PLAN_STEPS", 6),
        description="The maximum number of plan steps",
        ge=1,
    )
    MCP_URL: str = Field(os.getenv("MCP_URL", "http://slack-tool:8000"), description="Endpoint for an option MCP server")
    SERVICE_PORT: int = Field(os.getenv("SERVICE_URL", 8000), description="Port on which the service will run.")

    # auth variables
    ISSUER: Optional[str] = Field(
        os.getenv("ISSUER", None), 
        description="The issuer for incoming JWT tokens"
    )
    INTROSPECTION_ENDPOINT: Optional[str] = Field(
        os.getenv("INTROSPECTION_ENDPOINT", None),
        description="The introspection endpoint for incoming JWT tokens"
    )
    AUDIENCE: Optional[str] = Field(
        os.getenv("AUDIENCE", None),
        description="Expected audience value during resource validation"
    )
    CLIENT_ID: Optional[str] = Field(
        os.getenv("CLIENT_NAME", None),
        description="Client ID of the agent in the OAuth server"
    )
    CLIENT_SECRET: Optional[str] = Field( # Note it is assumed the app is a confidential app
        os.getenv("CLIENT_SECRET", None),
        description="Client secret of the agent in the OAuth server"
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
