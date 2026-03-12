"""Tests for currency_converter agent — models and exchange rate tool (isolated from heavy deps)."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock heavy dependencies before importing
for mod in [
    "langchain_core",
    "langchain_core.messages",
    "langchain_core.tools",
    "langchain_google_genai",
    "langchain_openai",
    "langgraph",
    "langgraph.checkpoint",
    "langgraph.checkpoint.memory",
    "langgraph.prebuilt",
]:
    sys.modules.setdefault(mod, MagicMock())

# Restore the real @tool decorator as identity so get_exchange_rate stays callable
mock_tool = MagicMock(side_effect=lambda *a, **kw: (lambda f: f) if not a else a[0])
sys.modules["langchain_core.tools"].tool = mock_tool

from app.agent import Configuration, ResponseFormat, get_exchange_rate


class TestConfiguration:
    """Test currency agent configuration defaults."""

    def test_default_model(self):
        config = Configuration()
        assert config.llm_model == "gpt-4o"

    def test_default_api_base_empty(self):
        config = Configuration()
        assert config.llm_api_base == ""


class TestResponseFormat:
    """Test the ResponseFormat Pydantic model."""

    def test_default_status(self):
        resp = ResponseFormat(message="hello")
        assert resp.status == "input_required"

    def test_completed_status(self):
        resp = ResponseFormat(status="completed", message="done")
        assert resp.status == "completed"

    def test_error_status(self):
        resp = ResponseFormat(status="error", message="fail")
        assert resp.status == "error"

    def test_invalid_status_rejected(self):
        with pytest.raises(Exception):
            ResponseFormat(status="unknown", message="bad")


class TestGetExchangeRate:
    """Test the get_exchange_rate tool function."""

    @patch("app.agent.httpx.get")
    def test_successful_exchange_rate(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "amount": 1.0,
            "base": "USD",
            "date": "2025-01-01",
            "rates": {"EUR": 0.85},
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = get_exchange_rate(currency_from="USD", currency_to="EUR")
        assert "rates" in result
        assert result["rates"]["EUR"] == 0.85

    @patch("app.agent.httpx.get")
    def test_invalid_api_response(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": "no rates"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = get_exchange_rate(currency_from="USD", currency_to="XYZ")
        assert "error" in result

    @patch("app.agent.httpx.get")
    def test_http_error(self, mock_get):
        import httpx

        mock_get.side_effect = httpx.HTTPError("Connection failed")

        result = get_exchange_rate(currency_from="USD", currency_to="EUR")
        assert "error" in result
        assert "API request failed" in result["error"]
