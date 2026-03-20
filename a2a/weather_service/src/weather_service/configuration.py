import logging

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

_PLACEHOLDER_KEYS = {"dummy", "changeme", "your-api-key-here", ""}

# API bases that are known to accept placeholder/dummy keys (local LLMs)
_LOCAL_LLM_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}


class Configuration(BaseSettings):
    llm_model: str = "llama3.1"
    llm_api_base: str = "http://localhost:11434/v1"
    llm_api_key: str = "dummy"

    @property
    def is_local_llm(self) -> bool:
        """Check if the API base points to a local LLM (e.g. Ollama)."""
        from urllib.parse import urlparse

        parsed = urlparse(self.llm_api_base)
        hostname = parsed.hostname or ""
        return hostname in _LOCAL_LLM_HOSTS

    @property
    def has_valid_api_key(self) -> bool:
        """Check if the API key is usable.

        Local LLMs (Ollama, vLLM, etc.) accept any key including placeholders,
        so placeholder keys are only flagged when pointing at a remote API.
        """
        if self.is_local_llm:
            return True
        return self.llm_api_key.strip() not in _PLACEHOLDER_KEYS

    def log_warnings(self) -> None:
        """Log warnings about configuration issues at startup."""
        if not self.has_valid_api_key:
            logger.warning(
                "No LLM API key configured (set LLM_API_KEY env var). "
                "The weather agent will not be able to call the LLM."
            )
