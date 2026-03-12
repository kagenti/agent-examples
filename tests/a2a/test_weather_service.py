"""Tests for weather_service agent — configuration (isolated from heavy deps).

The weather_service.__init__ calls setup_observability() at import time, which
depends on the full opentelemetry SDK. We bypass this by pre-registering a mock
for the observability module and the weather_service package itself.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock

# Create a fake weather_service package to prevent __init__.py from executing
_fake_ws = ModuleType("weather_service")
_fake_ws.__path__ = []  # type: ignore[attr-defined]
sys.modules["weather_service"] = _fake_ws
sys.modules["weather_service.observability"] = MagicMock()

# Now import the configuration module directly — it only needs pydantic_settings
from importlib import import_module
import importlib.util

# Load configuration.py directly from its file path
import pathlib

_config_path = (
    pathlib.Path(__file__).parent.parent.parent
    / "a2a"
    / "weather_service"
    / "src"
    / "weather_service"
    / "configuration.py"
)
spec = importlib.util.spec_from_file_location("weather_service.configuration", _config_path)
_config_mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules["weather_service.configuration"] = _config_mod
spec.loader.exec_module(_config_mod)  # type: ignore[union-attr]

Configuration = _config_mod.Configuration


class TestConfiguration:
    """Test weather service configuration defaults."""

    def test_default_model(self):
        config = Configuration()
        assert config.llm_model == "llama3.1"

    def test_default_api_base(self):
        config = Configuration()
        assert config.llm_api_base == "http://localhost:11434/v1"

    def test_default_api_key(self):
        config = Configuration()
        assert config.llm_api_key == "dummy"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_API_BASE", "https://api.openai.com/v1")
        config = Configuration()
        assert config.llm_model == "gpt-4o"
        assert config.llm_api_base == "https://api.openai.com/v1"
