import json
import os
import jwt
import sys
from pydantic_settings import BaseSettings
from pydantic import model_validator
from pydantic import Field
from typing import Literal, Optional

def get_client_id_from_svid() -> Optional]str]:
    """
    Read the SVID JWT from file and extract the client ID from the "sub" claim.
    """
    # Read SVID JWT from file to get client ID
    jwt_file_path = "/opt/jwt_svid.token"

    content = None
    try:
        with open(jwt_file_path, "r") as file:
            content = file.read()
    except FileNotFoundError:
        print(f"SVID JWT file {jwt_file_path} not found.")
        return None

    if content is None or content.strip() == "":
        print(f"No content in SVID JWT file {jwt_file_path}.")
        return None

    try:
        decoded = jwt.decode(content, options={"verify_signature": False})
    except jwt.DecodeError:
        print(f"Failed to decode SVID JWT file {jwt_file_path}.")
        return None

    try:
        return decoded["sub"]
    except KeyError:
        print("SVID JWT is missing required "sub" claim.")
        return None

def get_client_secret_from_svid(secret_file_path) -> Optional[str]:
    try:
        with open(secret_file_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"CLIENT_SECRET file not found at {self.secret_file_path}")
        return None
    except Exception as e:
        print(f"Error reading CLIENT_SECRET file: {e}")
        return None

class Settings(BaseSettings):
    # static path for client secret file
    secret_file_path: str = "/shared/secret.txt"

    LOG_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        os.getenv("LOG_LEVEL", "INFO"),
        description="Application log level",       
    )
    TASK_MODEL_ID: str = Field(
        os.getenv("TASK_MODEL_ID", "ollama/ibm/granite4:latest"),
        description="The ID of the task model",
    )
    LLM_API_BASE: str = Field(
        os.getenv("LLM_API_BASE", "http://host.docker.internal:11434"),
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
    SERVICE_PORT: int = Field(os.getenv("SERVICE_PORT", 8000), description="Port on which the service will run.")
    GITHUB_TOKEN: Optional[str] = Field(os.getenv("GITHUB_TOKEN", None), description="If not using agent with authorization, the default Github token to use")

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
        os.getenv("AUDIENCE", get_client_id_from_svid()),
        description="Expected audience value during resource validation"
    )

    # auth variables for token exchange
    TOKEN_URL: Optional[str] = Field(
        os.getenv("TOKEN_URL", None),
        description="Token endpoint to obtain new access tokens"
    )
    CLIENT_ID: Optional[str] = Field(
        os.getenv("CLIENT_ID", get_client_id_from_svid()),
        description="Client ID to authenticate to OAuth server"
    )
    CLIENT_SECRET: Optional[str] = Field(
        os.getenv("CLIENT_SECRET", get_client_secret_from_svid(secret_file_path)),
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
