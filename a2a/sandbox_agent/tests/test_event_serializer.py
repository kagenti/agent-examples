"""Tests for the event serializer.

Validates:
  - LangGraphSerializer handles planner, reflector, reporter node events
  - Executor events are serialized like assistant events (tool_call / llm_response)
  - Tool events are serialized as tool_result
  - Unknown nodes produce llm_response fallback
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from sandbox_agent.event_serializer import LangGraphSerializer


def _make_msg(content: str = "", tool_calls: list | None = None, name: str | None = None) -> MagicMock:
    """Create a mock message with content, tool_calls, and name attributes."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    if name is not None:
        msg.name = name
    return msg


class TestSerializePlanner:
    """Planner events should emit plan type with step list."""

    def test_plan_with_steps(self) -> None:
        s = LangGraphSerializer()
        result = s.serialize("planner", {
            "plan": ["List files", "Read config"],
            "iteration": 1,
            "messages": [],
        })
        data = json.loads(result)
        assert data["type"] == "plan"
        assert data["plan"] == ["List files", "Read config"]
        assert data["iteration"] == 1

    def test_plan_with_message(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="Here's my plan")
        result = s.serialize("planner", {
            "plan": ["Step 1"],
            "iteration": 1,
            "messages": [msg],
        })
        data = json.loads(result)
        assert data["content"] == "Here's my plan"

    def test_plan_empty(self) -> None:
        s = LangGraphSerializer()
        result = s.serialize("planner", {
            "messages": [],
        })
        data = json.loads(result)
        assert data["type"] == "plan"
        assert data["plan"] == []


class TestSerializeReflector:
    """Reflector events should emit reflection type with done status."""

    def test_reflection_continue(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="continue")
        result = s.serialize("reflector", {
            "done": False,
            "current_step": 1,
            "step_results": ["result1"],
            "messages": [msg],
        })
        data = json.loads(result)
        assert data["type"] == "reflection"
        assert data["done"] is False
        assert data["current_step"] == 1

    def test_reflection_done(self) -> None:
        s = LangGraphSerializer()
        result = s.serialize("reflector", {
            "done": True,
            "current_step": 3,
            "messages": [],
        })
        data = json.loads(result)
        assert data["type"] == "reflection"
        assert data["done"] is True


class TestSerializeReporter:
    """Reporter events should emit llm_response with final answer."""

    def test_reporter_with_final_answer(self) -> None:
        s = LangGraphSerializer()
        result = s.serialize("reporter", {
            "final_answer": "All done!",
            "messages": [],
        })
        data = json.loads(result)
        assert data["type"] == "llm_response"
        assert data["content"] == "All done!"

    def test_reporter_falls_back_to_message(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="Final summary text")
        result = s.serialize("reporter", {
            "messages": [msg],
        })
        data = json.loads(result)
        assert data["type"] == "llm_response"
        assert "Final summary" in data["content"]


class TestSerializeExecutor:
    """Executor events should serialize like assistant (tool_call / llm_response)."""

    def test_executor_llm_response(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="I completed the step")
        result = s.serialize("executor", {"messages": [msg]})
        data = json.loads(result)
        assert data["type"] == "llm_response"
        assert "completed" in data["content"]

    def test_executor_tool_call(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(
            content="Let me run a command",
            tool_calls=[{"name": "shell", "args": {"command": "ls"}}],
        )
        result = s.serialize("executor", {"messages": [msg]})
        # Should contain both thinking text and tool call
        lines = result.strip().split("\n")
        assert len(lines) == 2
        thinking = json.loads(lines[0])
        tool_call = json.loads(lines[1])
        assert thinking["type"] == "llm_response"
        assert tool_call["type"] == "tool_call"
        assert tool_call["tools"][0]["name"] == "shell"


class TestSerializeToolResult:
    """Tool events should serialize as tool_result."""

    def test_tool_result(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="file1.txt\nfile2.txt", name="shell")
        result = s.serialize("tools", {"messages": [msg]})
        data = json.loads(result)
        assert data["type"] == "tool_result"
        assert data["name"] == "shell"
        assert "file1.txt" in data["output"]


class TestSerializeUnknownNode:
    """Unknown nodes should fall back to llm_response."""

    def test_unknown_node(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="some output")
        result = s.serialize("custom_node", {"messages": [msg]})
        data = json.loads(result)
        assert data["type"] == "llm_response"

    def test_empty_messages(self) -> None:
        s = LangGraphSerializer()
        result = s.serialize("custom_node", {"messages": []})
        data = json.loads(result)
        assert data["type"] == "llm_response"
        assert "custom_node" in data["content"]
