import json
import logging
import os
import sys
from pydantic_settings import BaseSettings
from pydantic import model_validator
from pydantic import Field
from typing import Literal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s: %(message)s')


class Settings(BaseSettings):
    TASK_MODEL_ID: str = Field(
        os.getenv("TASK_MODEL_ID", "granite3.3:8b"),
        description="The ID of the task model",
    )
    OPENAI_API_URL: str = Field(
        os.getenv("OPENAI_API_URL", "http://localhost:11434/v1"),
        description="The URL for OpenAI API",
    )
    OPENAI_API_KEY: str = Field(os.getenv("OPENAI_API_KEY", "ollama"), description="The key for OpenAI API")
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
    TAVILY_API_KEY: str = Field(os.getenv("TAVILY_API_KEY", ""), description="The key for Tavily API")
    MCP_ENDPOINT: str = Field(os.getenv("MCP_ENDPOINT", ""), description="Endpoint for an option MCP server")
    MCP_TRANSPORT: Literal["sse", "streamable_http"] = Field(
        os.getenv(
            "MCP_TRANSPORT",
            "streamable_http"),
            description="The transport method of your MCP server. sse or streamable_http")
    KEYCLOAK_URL: str = Field(os.getenv("KEYCLOAK_URL", "http://keycloak.localtest.me:8080"),
                              descripiton="URL of keycloak server")
    SERVICE_PORT: int = Field(os.getenv("SERVICE_URL", 8000), description="Port on which the service will run.")

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

    @model_validator(mode="after")
    def set_secondary_env(self) -> "Settings":
        if "TAVILY_API_KEY" not in os.environ:
            os.environ["TAVILY_API_KEY"] = str(self.TAVILY_API_KEY)

        return self


settings = Settings()  # type: ignore[call-arg]
