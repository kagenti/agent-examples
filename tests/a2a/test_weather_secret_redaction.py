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
        return logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg=msg, args=args, exc_info=None,
        )

    def test_redacts_bearer_token(self):
        record = self._make_record("Authorization: Bearer sk-abc123xyz789secret")
        self.filt.filter(record)
        assert "sk-abc123xyz789secret" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_bearer_case_insensitive(self):
        record = self._make_record("header: bearer my-secret-token-value")
        self.filt.filter(record)
        assert "my-secret-token-value" not in record.msg

    def test_preserves_non_secret_messages(self):
        record = self._make_record("Processing weather request for New York")
        self.filt.filter(record)
        assert record.msg == "Processing weather request for New York"

    def test_redacts_bearer_in_tuple_args(self):
        record = self._make_record("Header: %s", ("Bearer secret123",))
        self.filt.filter(record)
        assert "secret123" not in record.args[0]

    def test_redacts_bearer_in_dict_args(self):
        record = self._make_record("%(auth)s")
        record.args = {"auth": "Bearer secret123"}
        self.filt.filter(record)
        assert "secret123" not in record.args["auth"]

    def test_always_returns_true(self):
        record = self._make_record("Bearer secret123")
        assert self.filt.filter(record) is True

    def test_redacts_literal_configured_key(self, monkeypatch):
        rhoai_key = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
        monkeypatch.setenv("LLM_API_KEY", rhoai_key)
        filt = SecretRedactionFilter()
        record = self._make_record(f"Sending request with api-key={rhoai_key}")
        filt.filter(record)
        assert rhoai_key not in record.msg
        assert "[REDACTED]" in record.msg

    def test_literal_key_redaction_in_args(self, monkeypatch):
        rhoai_key = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
        monkeypatch.setenv("LLM_API_KEY", rhoai_key)
        filt = SecretRedactionFilter()
        record = self._make_record("key=%s", (rhoai_key,))
        filt.filter(record)
        assert rhoai_key not in record.args[0]

    def test_short_key_not_redacted(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "dummy")
        filt = SecretRedactionFilter()
        record = self._make_record("Using dummy config for testing dummy values")
        filt.filter(record)
        assert "dummy" in record.msg

    def test_no_crash_when_key_unset(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        filt = SecretRedactionFilter()
        record = self._make_record("Normal log message")
        filt.filter(record)
        assert record.msg == "Normal log message"


class TestConfigurationApiKeyValidation:
    """Test API key validation logic."""

    def test_dummy_key_with_remote_api_is_invalid(self, monkeypatch):
        monkeypatch.setenv("LLM_API_BASE", "https://api.openai.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "dummy")
        assert Configuration().has_valid_api_key is False

    def test_dummy_key_with_localhost_is_valid(self):
        config = Configuration()  # defaults: localhost + dummy
        assert config.has_valid_api_key is True

    def test_empty_key_with_remote_api_is_invalid(self, monkeypatch):
        monkeypatch.setenv("LLM_API_BASE", "https://api.openai.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "")
        assert Configuration().has_valid_api_key is False

    def test_placeholder_keys_with_remote_are_invalid(self, monkeypatch):
        monkeypatch.setenv("LLM_API_BASE", "https://api.openai.com/v1")
        for key in ["changeme", "your-api-key-here"]:
            monkeypatch.setenv("LLM_API_KEY", key)
            assert Configuration().has_valid_api_key is False, f"'{key}' should be invalid"

    def test_placeholder_keys_with_local_llm_are_valid(self, monkeypatch):
        monkeypatch.setenv("LLM_API_BASE", "http://localhost:11434/v1")
        for key in ["dummy", "changeme", ""]:
            monkeypatch.setenv("LLM_API_KEY", key)
            assert Configuration().has_valid_api_key is True

    def test_real_key_is_valid(self, monkeypatch):
        monkeypatch.setenv("LLM_API_BASE", "https://api.openai.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "sk-proj-realkey123")
        assert Configuration().has_valid_api_key is True

    def test_rhoai_maas_key_is_valid(self, monkeypatch):
        monkeypatch.setenv(
            "LLM_API_BASE",
            "https://model--maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1",
        )
        monkeypatch.setenv("LLM_API_KEY", "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6")
        assert Configuration().has_valid_api_key is True

    def test_127_is_local(self, monkeypatch):
        monkeypatch.setenv("LLM_API_BASE", "http://127.0.0.1:8080/v1")
        assert Configuration().has_valid_api_key is True
