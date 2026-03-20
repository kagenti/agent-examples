import logging

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

_PLACEHOLDER_KEYS = {"dummy", "changeme", "your-api-key-here", ""}


class Configuration(BaseSettings):
    llm_model: str = "llama3.1"
    llm_api_base: str = "http://localhost:11434/v1"
    llm_api_key: str = "dummy"

    @property
    def has_valid_api_key(self) -> bool:
        """Check if the API key appears to be a real (non-placeholder) value."""
        return self.llm_api_key.strip() not in _PLACEHOLDER_KEYS

    def log_warnings(self) -> None:
        """Log warnings about configuration issues at startup."""
        if not self.has_valid_api_key:
            logger.warning(
                "No LLM API key configured (set LLM_API_KEY env var). "
                "The weather agent will not be able to call the LLM."
            )
