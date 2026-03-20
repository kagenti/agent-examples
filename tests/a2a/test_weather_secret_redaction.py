"""Tests for secret redaction and API key validation in the weather service.

Loads agent.py and configuration.py in isolation (same approach as
test_weather_service.py) to avoid pulling in heavy deps like opentelemetry.
"""

import importlib.util
import logging
import pathlib
import sys
from types import ModuleType
from unittest.mock import MagicMock

# --- Isolation setup (must happen before any weather_service imports) ---
_fake_ws = ModuleType("weather_service")
_fake_ws.__path__ = []  # type: ignore[attr-defined]
sys.modules.setdefault("weather_service", _fake_ws)
sys.modules.setdefault("weather_service.observability", MagicMock())

_BASE = pathlib.Path(__file__).parent.parent.parent / "a2a" / "weather_service" / "src" / "weather_service"


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_config_mod = _load_module("weather_service.configuration", _BASE / "configuration.py")

# Mock modules that agent.py imports but we don't need
for mod_name in [
    "uvicorn",
    "langchain_core",
    "langchain_core.messages",
    "starlette",
    "starlette.middleware",
    "starlette.middleware.base",
    "starlette.routing",
    "a2a",
    "a2a.server",
    "a2a.server.agent_execution",
    "a2a.server.apps",
    "a2a.server.events",
    "a2a.server.events.event_queue",
    "a2a.server.request_handlers",
    "a2a.server.tasks",
    "a2a.types",
    "a2a.utils",
    "weather_service.graph",
]:
    sys.modules.setdefault(mod_name, MagicMock())

_agent_mod = _load_module("weather_service.agent", _BASE / "agent.py")

Configuration = _config_mod.Configuration
SecretRedactionFilter = _agent_mod.SecretRedactionFilter


# --- Tests ---


class TestSecretRedactionFilter:
    """Test the logging filter that redacts Bearer tokens and API keys."""

    def setup_method(self):
        self.filt = SecretRedactionFilter()

    def _make_record(self, msg: str, args=None) -> logging.LogRecord:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=args,
            exc_info=None,
        )
        return record

    def test_redacts_bearer_token_in_msg(self):
        record = self._make_record("Authorization: Bearer sk-abc123xyz789secret")
        self.filt.filter(record)
        assert "sk-abc123xyz789secret" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_redacts_bearer_token_case_insensitive(self):
        record = self._make_record("header: bearer my-secret-token-value")
        self.filt.filter(record)
        assert "my-secret-token-value" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_redacts_openai_api_key_pattern(self):
        record = self._make_record("Using key sk-proj1234567890abcdefghijklmnop")
        self.filt.filter(record)
        assert "1234567890abcdefghijklmnop" not in record.msg
        assert "sk-pro...[REDACTED]" in record.msg

    def test_preserves_non_secret_messages(self):
        record = self._make_record("Processing weather request for New York")
        self.filt.filter(record)
        assert record.msg == "Processing weather request for New York"

    def test_redacts_bearer_in_args(self):
        record = self._make_record("Header: %s", ("Bearer sk-abc123xyz789secret",))
        self.filt.filter(record)
        assert "sk-abc123xyz789secret" not in record.args[0]
        assert "[REDACTED]" in record.args[0]

    def test_always_returns_true(self):
        """Filter should never suppress log records, only redact content."""
        record = self._make_record("Bearer secret123")
        assert self.filt.filter(record) is True


class TestConfigurationApiKeyValidation:
    """Test API key validation logic."""

    def test_dummy_key_with_remote_api_is_invalid(self, monkeypatch):
        """Dummy key pointing at OpenAI should be flagged."""
        monkeypatch.setenv("LLM_API_BASE", "https://api.openai.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "dummy")
        config = Configuration()
        assert config.has_valid_api_key is False

    def test_dummy_key_with_ollama_is_valid(self):
        """Default config (Ollama on localhost) should work with dummy key."""
        config = Configuration()
        assert config.is_local_llm is True
        assert config.has_valid_api_key is True

    def test_empty_key_with_remote_api_is_invalid(self, monkeypatch):
        monkeypatch.setenv("LLM_API_BASE", "https://api.openai.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "")
        config = Configuration()
        assert config.has_valid_api_key is False

    def test_placeholder_keys_with_remote_api_are_invalid(self, monkeypatch):
        monkeypatch.setenv("LLM_API_BASE", "https://api.openai.com/v1")
        for placeholder in ["changeme", "your-api-key-here"]:
            monkeypatch.setenv("LLM_API_KEY", placeholder)
            config = Configuration()
            assert config.has_valid_api_key is False, f"'{placeholder}' should be invalid"

    def test_placeholder_keys_with_local_llm_are_valid(self, monkeypatch):
        """Local LLMs (Ollama, vLLM) accept any key — don't block them."""
        monkeypatch.setenv("LLM_API_BASE", "http://localhost:11434/v1")
        for placeholder in ["dummy", "changeme", ""]:
            monkeypatch.setenv("LLM_API_KEY", placeholder)
            config = Configuration()
            assert config.has_valid_api_key is True, f"'{placeholder}' with local LLM should be valid"

    def test_real_key_is_valid(self, monkeypatch):
        monkeypatch.setenv("LLM_API_BASE", "https://api.openai.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "sk-proj-realkey123")
        config = Configuration()
        assert config.has_valid_api_key is True

    def test_is_local_llm_with_127(self, monkeypatch):
        monkeypatch.setenv("LLM_API_BASE", "http://127.0.0.1:8080/v1")
        config = Configuration()
        assert config.is_local_llm is True

    def test_is_not_local_llm_with_remote_host(self, monkeypatch):
        monkeypatch.setenv("LLM_API_BASE", "https://api.openai.com/v1")
        config = Configuration()
        assert config.is_local_llm is False

    def test_log_warnings_with_dummy_key_remote(self, monkeypatch, caplog):
        monkeypatch.setenv("LLM_API_BASE", "https://api.openai.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "dummy")
        config = Configuration()
        with caplog.at_level(logging.WARNING):
            config.log_warnings()
        assert "No LLM API key configured" in caplog.text

    def test_no_warning_with_ollama_dummy_key(self, caplog):
        """Default Ollama config should NOT warn about the dummy key."""
        config = Configuration()
        with caplog.at_level(logging.WARNING):
            config.log_warnings()
        assert "No LLM API key configured" not in caplog.text

    def test_log_warnings_with_real_key(self, monkeypatch, caplog):
        monkeypatch.setenv("LLM_API_BASE", "https://api.openai.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "sk-proj-realkey123")
        config = Configuration()
        with caplog.at_level(logging.WARNING):
            config.log_warnings()
        assert "No LLM API key configured" not in caplog.text
