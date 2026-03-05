"""Tests for the event serializer.

Validates:
  - LangGraphSerializer includes loop_id in all reasoning loop events
  - Planner emits plan type with steps list
  - Executor emits plan_step + tool_call/llm_response events
  - Reflector emits reflection with assessment
  - Reporter emits llm_response with final answer
  - Tool results include loop_id and step
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


def _parse_lines(result: str) -> list[dict]:
    """Parse newline-delimited JSON events."""
    return [json.loads(line) for line in result.strip().split("\n") if line.strip()]


class TestSerializePlanner:
    """Planner events should emit plan type with steps and loop_id."""

    def test_plan_with_steps(self) -> None:
        s = LangGraphSerializer()
        result = s.serialize("planner", {
            "plan": ["List files", "Read config"],
            "iteration": 1,
            "messages": [],
        })
        data = json.loads(result)
        assert data["type"] == "plan"
        assert data["steps"] == ["List files", "Read config"]
        assert data["iteration"] == 1
        assert "loop_id" in data

    def test_plan_includes_loop_id(self) -> None:
        s = LangGraphSerializer(loop_id="test-loop")
        result = s.serialize("planner", {
            "plan": ["Step 1"],
            "iteration": 1,
            "messages": [],
        })
        data = json.loads(result)
        assert data["loop_id"] == "test-loop"

    def test_plan_empty(self) -> None:
        s = LangGraphSerializer()
        result = s.serialize("planner", {"messages": []})
        data = json.loads(result)
        assert data["type"] == "plan"
        assert data["steps"] == []


class TestSerializeReflector:
    """Reflector events should emit reflection with loop_id and assessment."""

    def test_reflection_continue(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="continue")
        result = s.serialize("reflector", {
            "done": False,
            "current_step": 1,
            "messages": [msg],
        })
        data = json.loads(result)
        assert data["type"] == "reflection"
        assert data["done"] is False
        assert data["current_step"] == 1
        assert "loop_id" in data
        assert data["assessment"] == "continue"

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
    """Reporter events should emit llm_response with loop_id."""

    def test_reporter_with_final_answer(self) -> None:
        s = LangGraphSerializer()
        result = s.serialize("reporter", {
            "final_answer": "All done!",
            "messages": [],
        })
        data = json.loads(result)
        assert data["type"] == "llm_response"
        assert data["content"] == "All done!"
        assert "loop_id" in data

    def test_reporter_falls_back_to_message(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="Final summary text")
        result = s.serialize("reporter", {"messages": [msg]})
        data = json.loads(result)
        assert data["type"] == "llm_response"
        assert "Final summary" in data["content"]


class TestSerializeExecutor:
    """Executor events emit plan_step + tool_call/llm_response with loop_id."""

    def test_executor_tool_call_emits_three_events(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(
            content="Let me run a command",
            tool_calls=[{"name": "shell", "args": {"command": "ls"}}],
        )
        result = s.serialize("executor", {"messages": [msg]})
        events = _parse_lines(result)
        # plan_step, llm_response (thinking), tool_call
        assert len(events) == 3
        assert events[0]["type"] == "plan_step"
        assert events[0]["loop_id"] == s._loop_id
        assert events[1]["type"] == "llm_response"
        assert events[2]["type"] == "tool_call"
        assert events[2]["tools"][0]["name"] == "shell"

    def test_executor_llm_only_emits_two_events(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="I completed the step")
        result = s.serialize("executor", {"messages": [msg]})
        events = _parse_lines(result)
        # plan_step + llm_response
        assert len(events) == 2
        assert events[0]["type"] == "plan_step"
        assert events[1]["type"] == "llm_response"
        assert "completed" in events[1]["content"]


class TestSerializeToolResult:
    """Tool events should serialize as tool_result with loop_id."""

    def test_tool_result(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="file1.txt\nfile2.txt", name="shell")
        result = s.serialize("tools", {"messages": [msg]})
        data = json.loads(result)
        assert data["type"] == "tool_result"
        assert data["name"] == "shell"
        assert "file1.txt" in data["output"]
        assert "loop_id" in data

    def test_tool_result_includes_step(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="output", name="file_read")
        result = s.serialize("tools", {"messages": [msg]})
        data = json.loads(result)
        assert "step" in data


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
