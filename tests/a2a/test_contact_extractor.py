"""Tests for contact_extractor agent — models and helper functions (isolated from heavy deps)."""

import sys
from unittest.mock import MagicMock

# Mock the marvin dependency before importing
sys.modules.setdefault("marvin", MagicMock())

from agent import ExtractionOutcome, ExtractorAgent, TextPart, _to_text_part
from pydantic import BaseModel


class TestTextPart:
    """Test TextPart model."""

    def test_text_part_defaults(self):
        part = TextPart(text="hello")
        assert part.type == "text"
        assert part.text == "hello"
        assert part.metadata is None

    def test_text_part_with_metadata(self):
        part = TextPart(text="hello", metadata={"key": "value"})
        assert part.metadata == {"key": "value"}


class TestToTextPart:
    """Test _to_text_part helper."""

    def test_creates_text_part(self):
        result = _to_text_part("test message")
        assert isinstance(result, TextPart)
        assert result.text == "test message"
        assert result.type == "text"


class TestExtractionOutcome:
    """Test ExtractionOutcome generic model."""

    def test_with_simple_type(self):
        class ContactInfo(BaseModel):
            name: str
            email: str

        outcome = ExtractionOutcome[ContactInfo](
            extracted_data=ContactInfo(name="Alice", email="alice@example.com"),
            summary="Extracted contact info for Alice",
        )
        assert outcome.extracted_data.name == "Alice"
        assert outcome.summary == "Extracted contact info for Alice"


class TestExtractorAgent:
    """Test ExtractorAgent initialization."""

    def test_supported_content_types(self):
        assert "text" in ExtractorAgent.SUPPORTED_CONTENT_TYPES
        assert "application/json" in ExtractorAgent.SUPPORTED_CONTENT_TYPES

    def test_init(self):
        agent = ExtractorAgent(instructions="Be helpful", result_type=str)
        assert agent.instructions == "Be helpful"
        assert agent.result_type is str
