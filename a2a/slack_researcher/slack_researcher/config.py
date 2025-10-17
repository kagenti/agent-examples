import json
import logging
import os
import jwt
from pydantic_settings import BaseSettings
from pydantic import model_validator
from pydantic import Field
from typing import Literal, Optional

def get_client_id() -> str:
    """
    Read the SVID JWT from file and extract the client ID from the "sub" claim.
    """
    # Read SVID JWT from file to get client ID
    jwt_file_path = "/opt/jwt_svid.token"
    try:
        with open(jwt_file_path, "r") as file:
            content = file.read()

    except FileNotFoundError:
        print(f"Error: The file {jwt_file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    if content is None or content.strip() == "":
        raise Exception(f"No content read from SVID JWT.")

    decoded = jwt.decode(content, options={"verify_signature": False})
    if "sub" not in decoded:
        raise Exception('SVID JWT does not contain a "sub" claim.')
    return decoded["sub"]

class Settings(BaseSettings):
    # static path for client secret file
    secret_file_path: str = "/shared/secret.txt"

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

    # auth variables for token validation
    ISSUER: Optional[str] = Field(
        os.getenv("ISSUER", None),
        description="The issuer for incoming JWT tokens"
    )
    JWKS_URI: Optional[str] = Field(
        os.getenv("JWKS_URI", None),
        description="Endpoint to obtain JWKS from auth server"
    )

    # auth variables for token exchange
    TOKEN_URL: Optional[str] = Field(
        os.getenv("TOKEN_URL", None),
        description="Token endpoint to obtain new access tokens"
    )
    CLIENT_ID: Optional[str] = Field(
        get_client_id(),
        description="Client ID to authenticate to OAuth server"
    )
    CLIENT_SECRET: Optional[str] = Field(
        None,
        description="Client secret to authenticate to OAuth server"
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

        # --- Load CLIENT_SECRET from file ---
        try:
            with open(self.secret_file_path, "r") as f:
                self.CLIENT_SECRET = f.read().strip()
        except FileNotFoundError:
            logging.warning(f"CLIENT_SECRET file not found at {self.secret_file_path}")
        except Exception as e:
            logging.error(f"Error reading CLIENT_SECRET file: {e}")

        return self

settings = Settings()  # type: ignore[call-arg]