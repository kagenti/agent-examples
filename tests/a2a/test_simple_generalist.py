"""Tests for simple_generalist agent — settings validation."""

from simple_generalist.config.settings import Settings


class TestSettings:
    """Test Settings configuration and validation."""

    def test_default_log_level(self, monkeypatch):
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        settings = Settings()
        assert settings.LOG_LEVEL == "INFO"

    def test_default_host_and_port(self, monkeypatch):
        monkeypatch.delenv("A2A_HOST", raising=False)
        monkeypatch.delenv("A2A_PORT", raising=False)
        settings = Settings()
        assert settings.A2A_HOST == "0.0.0.0"
        assert settings.A2A_PORT == 8000

    def test_extra_headers_from_json_string(self, monkeypatch):
        monkeypatch.setenv("EXTRA_HEADERS", '{"X-Custom": "value"}')
        settings = Settings()
        assert settings.EXTRA_HEADERS == {"X-Custom": "value"}

    def test_extra_headers_empty_json_object(self, monkeypatch):
        monkeypatch.setenv("EXTRA_HEADERS", "{}")
        settings = Settings()
        assert settings.EXTRA_HEADERS == {}

    def test_extra_headers_none(self, monkeypatch):
        monkeypatch.delenv("EXTRA_HEADERS", raising=False)
        settings = Settings()
        assert settings.EXTRA_HEADERS == {}

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "llama3.2")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
        monkeypatch.setenv("MAX_ITERATIONS", "50")
        settings = Settings()
        assert settings.LLM_MODEL == "llama3.2"
        assert settings.LLM_TEMPERATURE == 0.7
        assert settings.MAX_ITERATIONS == 50
