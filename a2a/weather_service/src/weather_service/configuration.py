from urllib.parse import urlparse

from pydantic_settings import BaseSettings

_PLACEHOLDER_KEYS = {"dummy", "changeme", "your-api-key-here", ""}


class Configuration(BaseSettings):
    llm_model: str = "llama3.1"
    llm_api_base: str = "http://localhost:11434/v1"
    llm_api_key: str = "dummy"

    @property
    def has_valid_api_key(self) -> bool:
        """Placeholder keys are only invalid for remote APIs.

        Local LLMs (Ollama, vLLM) accept any key, so we skip validation
        when the API base points to localhost.
        """
        host = urlparse(self.llm_api_base).hostname or ""
        # Local LLMs: localhost, docker host aliases, cluster-internal services
        if host in {"localhost", "127.0.0.1", "0.0.0.0", "dockerhost", "host.docker.internal"}:
            return True
        # Cluster-internal services (*.svc.cluster.local) accept any key
        if host.endswith(".svc.cluster.local"):
            return True
        return self.llm_api_key.strip() not in _PLACEHOLDER_KEYS
