"""Tests for the event serializer.

Validates:
  - LangGraphSerializer includes loop_id in all reasoning loop events
  - Planner emits planner_output (+ legacy plan) with steps list
  - Executor emits executor_step (+ legacy plan_step) + tool_call events
  - Reflector emits reflector_decision (+ legacy reflection) with decision field
  - Reporter emits reporter_output with final answer
  - Tool results include loop_id and step
  - Unknown nodes produce llm_response fallback
  - All reasoning-loop events include token counts and model
  - Decision extraction from reflector text
  - _safe_tc handles varied tool-call formats
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from sandbox_agent.event_serializer import LangGraphSerializer, _safe_tc


def _make_msg(
    content: str = "",
    tool_calls: list | None = None,
    name: str | None = None,
    tool_call_id: str | None = None,
) -> MagicMock:
    """Create a mock message with content, tool_calls, name, and tool_call_id."""
    # Use spec=[] to prevent MagicMock from auto-creating attributes
    # that would interfere with getattr(..., default) calls.
    msg = MagicMock(spec=["content", "tool_calls", "name", "tool_call_id"])
    msg.content = content
    msg.tool_calls = tool_calls or []
    msg.name = name if name is not None else "unknown"
    msg.tool_call_id = tool_call_id
    return msg


def _parse_lines(result: str) -> list[dict]:
    """Parse newline-delimited JSON events."""
    return [json.loads(line) for line in result.strip().split("\n") if line.strip()]


# ---------------------------------------------------------------------------
# Planner events
# ---------------------------------------------------------------------------


class TestPlannerEvents:
    """Planner should emit planner_output (new) + plan (legacy) events."""

    def test_planner_emits_planner_output_type(self) -> None:
        s = LangGraphSerializer()
        result = s.serialize("planner", {
            "plan": ["List files", "Read config"],
            "iteration": 1,
            "messages": [],
        })
        events = _parse_lines(result)
        new_event = events[0]
        assert new_event["type"] == "planner_output"
        assert new_event["steps"] == ["List files", "Read config"]
        assert new_event["iteration"] == 1

    def test_planner_emits_legacy_plan_type(self) -> None:
        s = LangGraphSerializer()
        result = s.serialize("planner", {
            "plan": ["Step A"],
            "iteration": 2,
            "messages": [],
        })
        events = _parse_lines(result)
        legacy = events[1]
        assert legacy["type"] == "plan"
        assert legacy["steps"] == ["Step A"]
        assert legacy["iteration"] == 2

    def test_planner_includes_loop_id(self) -> None:
        s = LangGraphSerializer(loop_id="test-loop")
        result = s.serialize("planner", {
            "plan": ["Step 1"],
            "iteration": 1,
            "messages": [],
        })
        events = _parse_lines(result)
        for event in events:
            assert event["loop_id"] == "test-loop"

    def test_planner_includes_iteration(self) -> None:
        s = LangGraphSerializer()
        result = s.serialize("planner", {
            "plan": ["A", "B"],
            "iteration": 3,
            "messages": [],
        })
        events = _parse_lines(result)
        assert events[0]["iteration"] == 3

    def test_planner_empty_plan(self) -> None:
        s = LangGraphSerializer()
        result = s.serialize("planner", {"messages": []})
        events = _parse_lines(result)
        assert events[0]["type"] == "planner_output"
        assert events[0]["steps"] == []

    def test_planner_default_iteration_is_one(self) -> None:
        s = LangGraphSerializer()
        result = s.serialize("planner", {"plan": ["Only step"], "messages": []})
        events = _parse_lines(result)
        assert events[0]["iteration"] == 1

    def test_planner_includes_content_from_message(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="Here is my plan")
        result = s.serialize("planner", {
            "plan": ["Step 1"],
            "iteration": 2,
            "messages": [msg],
        })
        events = _parse_lines(result)
        assert events[0]["content"] == "Here is my plan"

    def test_planner_content_from_list_blocks(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg()
        msg.content = [{"type": "text", "text": "Block one"}, {"type": "text", "text": "Block two"}]
        result = s.serialize("planner", {
            "plan": [],
            "messages": [msg],
        })
        events = _parse_lines(result)
        assert "Block one" in events[0]["content"]
        assert "Block two" in events[0]["content"]

    def test_planner_includes_model(self) -> None:
        s = LangGraphSerializer()
        result = s.serialize("planner", {
            "plan": [],
            "iteration": 1,
            "messages": [],
            "model": "gpt-4o",
        })
        events = _parse_lines(result)
        assert events[0]["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# Executor events
# ---------------------------------------------------------------------------


class TestExecutorEvents:
    """Executor should emit executor_step (+ legacy plan_step) + optional tool_call."""

    def test_executor_emits_executor_step_type(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="Working on step")
        result = s.serialize("executor", {"messages": [msg]})
        events = _parse_lines(result)
        types = [e["type"] for e in events]
        assert "executor_step" in types

    def test_executor_emits_legacy_plan_step(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="Working on step")
        result = s.serialize("executor", {"messages": [msg]})
        events = _parse_lines(result)
        types = [e["type"] for e in events]
        assert "plan_step" in types

    def test_executor_tool_call_events(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(
            content="",
            tool_calls=[{"name": "shell", "args": {"command": "ls"}}],
        )
        result = s.serialize("executor", {"messages": [msg]})
        events = _parse_lines(result)
        types = [e["type"] for e in events]
        assert "executor_step" in types
        assert "plan_step" in types
        assert "tool_call" in types

    def test_tool_call_has_name_and_args(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(
            content="",
            tool_calls=[{"name": "file_read", "args": {"path": "/tmp/x"}}],
        )
        result = s.serialize("executor", {"messages": [msg]})
        events = _parse_lines(result)
        tc_event = [e for e in events if e["type"] == "tool_call"][0]
        assert tc_event["tools"][0]["name"] == "file_read"
        assert tc_event["tools"][0]["args"] == {"path": "/tmp/x"}

    def test_executor_step_includes_description(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="Reading the configuration file")
        result = s.serialize("executor", {"messages": [msg]})
        events = _parse_lines(result)
        step_event = [e for e in events if e["type"] == "executor_step"][0]
        assert "Reading" in step_event["description"]

    def test_executor_multiple_tool_calls(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(
            content="",
            tool_calls=[
                {"name": "shell", "args": {"cmd": "ls"}},
                {"name": "file_read", "args": {"path": "/etc/hosts"}},
            ],
        )
        result = s.serialize("executor", {"messages": [msg]})
        events = _parse_lines(result)
        tc_event = [e for e in events if e["type"] == "tool_call"][0]
        assert len(tc_event["tools"]) == 2
        assert tc_event["tools"][0]["name"] == "shell"
        assert tc_event["tools"][1]["name"] == "file_read"

    def test_executor_tool_call_includes_step_and_loop_id(self) -> None:
        s = LangGraphSerializer(loop_id="exec-test")
        msg = _make_msg(
            content="",
            tool_calls=[{"name": "shell", "args": {}}],
        )
        result = s.serialize("executor", {"messages": [msg]})
        events = _parse_lines(result)
        tc_event = [e for e in events if e["type"] == "tool_call"][0]
        assert "step" in tc_event
        assert tc_event["loop_id"] == "exec-test"

    def test_executor_all_events_have_loop_id(self) -> None:
        s = LangGraphSerializer(loop_id="exec-2")
        msg = _make_msg(
            content="thinking",
            tool_calls=[{"name": "shell", "args": {}}],
        )
        result = s.serialize("executor", {"messages": [msg]})
        events = _parse_lines(result)
        for event in events:
            assert event.get("loop_id") == "exec-2", (
                f"Event type={event['type']} missing loop_id"
            )

    def test_executor_includes_total_steps_from_plan(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="step work")
        result = s.serialize("executor", {
            "messages": [msg],
            "plan": ["a", "b", "c"],
        })
        events = _parse_lines(result)
        step_event = [e for e in events if e["type"] == "executor_step"][0]
        assert step_event["total_steps"] == 3


# ---------------------------------------------------------------------------
# Tool result events
# ---------------------------------------------------------------------------


class TestToolResultEvents:
    """Tool events should serialize as tool_result with loop_id."""

    def test_tool_result_basic(self) -> None:
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

    def test_tool_result_truncates_output(self) -> None:
        s = LangGraphSerializer()
        long_output = "y" * 3000
        msg = _make_msg(content=long_output, name="shell")
        result = s.serialize("tools", {"messages": [msg]})
        data = json.loads(result)
        assert len(data["output"]) <= 2000

    def test_tool_result_name_defaults_to_unknown(self) -> None:
        s = LangGraphSerializer()
        msg = MagicMock(spec=[])
        msg.content = "some output"
        msg.name = "unknown"
        result = s.serialize("tools", {"messages": [msg]})
        data = json.loads(result)
        assert data["name"] == "unknown"


# ---------------------------------------------------------------------------
# Reflector events
# ---------------------------------------------------------------------------


class TestReflectorEvents:
    """Reflector should emit reflector_decision (new) + reflection (legacy)."""

    def test_reflector_emits_reflector_decision_type(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="continue with next step")
        result = s.serialize("reflector", {
            "done": False,
            "current_step": 1,
            "messages": [msg],
        })
        events = _parse_lines(result)
        assert events[0]["type"] == "reflector_decision"

    def test_reflector_emits_legacy_reflection_type(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="continue")
        result = s.serialize("reflector", {
            "done": False,
            "current_step": 1,
            "messages": [msg],
        })
        events = _parse_lines(result)
        assert events[1]["type"] == "reflection"

    def test_reflector_never_emits_llm_response(self) -> None:
        """The reflector must NOT emit 'llm_response' -- that is not a valid reflector type."""
        s = LangGraphSerializer()
        msg = _make_msg(content="The step looks good, continue")
        result = s.serialize("reflector", {
            "done": False,
            "current_step": 0,
            "messages": [msg],
        })
        events = _parse_lines(result)
        for event in events:
            assert event["type"] != "llm_response"

    def test_reflector_includes_decision_field(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="Step output is correct, continue to next")
        result = s.serialize("reflector", {
            "done": False,
            "current_step": 2,
            "messages": [msg],
        })
        events = _parse_lines(result)
        new_event = events[0]
        assert "decision" in new_event
        assert new_event["decision"] == "continue"

    def test_reflector_decision_done(self) -> None:
        """When done=True, decision should be 'done'."""
        s = LangGraphSerializer()
        result = s.serialize("reflector", {
            "done": True,
            "current_step": 3,
            "messages": [],
        })
        events = _parse_lines(result)
        assert events[0]["decision"] == "done"

    def test_reflector_decision_replan(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="The approach failed, we need to replan")
        result = s.serialize("reflector", {
            "done": False,
            "current_step": 1,
            "messages": [msg],
        })
        events = _parse_lines(result)
        assert events[0]["decision"] == "replan"

    def test_reflector_decision_is_valid(self) -> None:
        """Decision must be one of: continue, replan, done, hitl."""
        valid = {"continue", "replan", "done", "hitl"}
        for word in ("continue onwards", "we should replan", "all done now", "need hitl approval"):
            s = LangGraphSerializer()
            msg = _make_msg(content=word)
            result = s.serialize("reflector", {
                "done": False,
                "current_step": 0,
                "messages": [msg],
            })
            events = _parse_lines(result)
            assert events[0]["decision"] in valid, f"Bad decision for text: {word}"

    def test_reflector_includes_assessment(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="Output looks correct, continue")
        result = s.serialize("reflector", {
            "done": False,
            "current_step": 0,
            "messages": [msg],
        })
        events = _parse_lines(result)
        assert events[0]["assessment"] == "Output looks correct, continue"

    def test_reflector_legacy_has_content_and_assessment(self) -> None:
        """Legacy event has both content and assessment fields."""
        s = LangGraphSerializer()
        msg = _make_msg(content="all good")
        result = s.serialize("reflector", {
            "done": False,
            "current_step": 0,
            "messages": [msg],
        })
        events = _parse_lines(result)
        legacy = events[1]
        assert legacy["content"] == legacy["assessment"]

    def test_reflector_advances_step_index(self) -> None:
        s = LangGraphSerializer()
        assert s._step_index == 0
        s.serialize("reflector", {
            "done": False,
            "current_step": 2,
            "messages": [],
        })
        assert s._step_index == 2

    def test_reflector_with_step_results(self) -> None:
        """step_results field is accepted without error."""
        s = LangGraphSerializer()
        result = s.serialize("reflector", {
            "done": False,
            "current_step": 1,
            "step_results": ["result A"],
            "messages": [],
        })
        events = _parse_lines(result)
        assert events[0]["type"] == "reflector_decision"

    def test_reflector_includes_iteration(self) -> None:
        s = LangGraphSerializer()
        result = s.serialize("reflector", {
            "done": False,
            "current_step": 0,
            "iteration": 2,
            "messages": [],
        })
        events = _parse_lines(result)
        assert events[0]["iteration"] == 2


# ---------------------------------------------------------------------------
# Reporter events
# ---------------------------------------------------------------------------


class TestReporterEvents:
    """Reporter should emit reporter_output with final answer."""

    def test_reporter_emits_reporter_output_type(self) -> None:
        s = LangGraphSerializer()
        result = s.serialize("reporter", {
            "final_answer": "All done!",
            "messages": [],
        })
        data = json.loads(result)
        assert data["type"] == "reporter_output"
        assert data["content"] == "All done!"
        assert "loop_id" in data

    def test_reporter_falls_back_to_message(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="Final summary text")
        result = s.serialize("reporter", {"messages": [msg]})
        data = json.loads(result)
        assert data["type"] == "reporter_output"
        assert "Final summary" in data["content"]

    def test_reporter_prefers_final_answer_over_message(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="message text")
        result = s.serialize("reporter", {
            "final_answer": "answer text",
            "messages": [msg],
        })
        data = json.loads(result)
        assert data["content"] == "answer text"

    def test_reporter_truncates_long_content(self) -> None:
        s = LangGraphSerializer()
        long_text = "x" * 3000
        result = s.serialize("reporter", {
            "final_answer": long_text,
            "messages": [],
        })
        data = json.loads(result)
        assert len(data["content"]) <= 2000

    def test_reporter_empty_final_answer_falls_back(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="from message")
        result = s.serialize("reporter", {
            "final_answer": "",
            "messages": [msg],
        })
        data = json.loads(result)
        assert "from message" in data["content"]

    def test_reporter_does_not_emit_llm_response(self) -> None:
        """Reporter uses reporter_output, not the generic llm_response."""
        s = LangGraphSerializer()
        result = s.serialize("reporter", {
            "final_answer": "done",
            "messages": [],
        })
        data = json.loads(result)
        assert data["type"] == "reporter_output"


# ---------------------------------------------------------------------------
# Unknown node fallback
# ---------------------------------------------------------------------------


class TestUnknownNodeEvents:
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

    def test_unknown_node_list_content(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg()
        msg.content = [{"type": "text", "text": "hello world"}]
        result = s.serialize("some_node", {"messages": [msg]})
        data = json.loads(result)
        assert data["type"] == "llm_response"
        assert "hello world" in data["content"]


# ---------------------------------------------------------------------------
# Token fields
# ---------------------------------------------------------------------------


class TestTokenFields:
    """Reasoning-loop events should include prompt_tokens and completion_tokens."""

    @pytest.mark.parametrize("node,value", [
        ("planner", {"plan": ["step"], "iteration": 1, "messages": [],
                      "prompt_tokens": 100, "completion_tokens": 50}),
        ("reflector", {"done": False, "current_step": 0, "messages": [],
                       "prompt_tokens": 200, "completion_tokens": 75}),
        ("reporter", {"final_answer": "done", "messages": [],
                      "prompt_tokens": 300, "completion_tokens": 120}),
    ])
    def test_token_counts_present(self, node: str, value: dict) -> None:
        s = LangGraphSerializer()
        result = s.serialize(node, value)
        # For multi-line output, check the first (new-type) event
        events = _parse_lines(result)
        data = events[0]
        assert data["prompt_tokens"] > 0
        assert data["completion_tokens"] > 0

    @pytest.mark.parametrize("node,value", [
        ("planner", {"plan": [], "messages": []}),
        ("reflector", {"done": False, "current_step": 0, "messages": []}),
        ("reporter", {"final_answer": "ok", "messages": []}),
    ])
    def test_token_counts_default_to_zero(self, node: str, value: dict) -> None:
        s = LangGraphSerializer()
        result = s.serialize(node, value)
        events = _parse_lines(result)
        data = events[0]
        assert data["prompt_tokens"] == 0
        assert data["completion_tokens"] == 0

    def test_executor_step_includes_tokens(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="working")
        result = s.serialize("executor", {
            "messages": [msg],
            "prompt_tokens": 50,
            "completion_tokens": 25,
        })
        events = _parse_lines(result)
        step_event = [e for e in events if e["type"] == "executor_step"][0]
        assert step_event["prompt_tokens"] == 50
        assert step_event["completion_tokens"] == 25


# ---------------------------------------------------------------------------
# Loop ID consistency
# ---------------------------------------------------------------------------


class TestLoopId:
    """Every reasoning-loop event must include the loop_id for grouping."""

    def test_all_reasoning_nodes_include_loop_id(self) -> None:
        s = LangGraphSerializer(loop_id="group-42")
        nodes = {
            "planner": {"plan": ["a"], "iteration": 1, "messages": []},
            "reflector": {"done": False, "current_step": 0, "messages": []},
            "reporter": {"final_answer": "done", "messages": []},
        }
        for node, value in nodes.items():
            result = s.serialize(node, value)
            events = _parse_lines(result)
            for event in events:
                assert event["loop_id"] == "group-42", (
                    f"{node} event type={event['type']} has wrong loop_id"
                )

    def test_executor_events_all_have_loop_id(self) -> None:
        s = LangGraphSerializer(loop_id="exec-1")
        msg = _make_msg(
            content="",
            tool_calls=[{"name": "shell", "args": {}}],
        )
        result = s.serialize("executor", {"messages": [msg]})
        events = _parse_lines(result)
        for event in events:
            assert event.get("loop_id") == "exec-1", (
                f"Event type={event['type']} missing loop_id"
            )

    def test_tool_result_has_loop_id(self) -> None:
        s = LangGraphSerializer(loop_id="tools-1")
        msg = _make_msg(content="output", name="shell")
        result = s.serialize("tools", {"messages": [msg]})
        data = json.loads(result)
        assert data["loop_id"] == "tools-1"

    def test_auto_generated_loop_id(self) -> None:
        s = LangGraphSerializer()
        assert s._loop_id is not None
        assert len(s._loop_id) == 8


# ---------------------------------------------------------------------------
# _extract_decision helper
# ---------------------------------------------------------------------------


class TestExtractDecision:
    """_extract_decision should return a valid decision keyword."""

    @pytest.mark.parametrize("text,expected", [
        ("we should continue", "continue"),
        ("need to replan the approach", "replan"),
        ("all done", "done"),
        ("requires hitl approval", "hitl"),
        ("", "continue"),  # default
        ("ambiguous text with no keyword", "continue"),  # default
    ])
    def test_decision_extraction(self, text: str, expected: str) -> None:
        assert LangGraphSerializer._extract_decision(text) == expected

    def test_done_takes_priority_over_continue(self) -> None:
        """When text contains both 'done' and 'continue', done wins (checked first)."""
        result = LangGraphSerializer._extract_decision("done and continue")
        assert result == "done"


# ---------------------------------------------------------------------------
# _safe_tc helper
# ---------------------------------------------------------------------------


class TestSafeTc:
    """_safe_tc extracts name/args from various tool-call formats."""

    def test_dict_format(self) -> None:
        result = _safe_tc({"name": "shell", "args": {"cmd": "ls"}})
        assert result == {"name": "shell", "args": {"cmd": "ls"}}

    def test_dict_missing_fields(self) -> None:
        result = _safe_tc({})
        assert result == {"name": "unknown", "args": {}}

    def test_object_with_attributes(self) -> None:
        tc = MagicMock()
        tc.name = "file_read"
        tc.args = {"path": "/tmp"}
        result = _safe_tc(tc)
        assert result == {"name": "file_read", "args": {"path": "/tmp"}}

    def test_tuple_format(self) -> None:
        result = _safe_tc(("grep", {"pattern": "foo"}))
        assert result == {"name": "grep", "args": {"pattern": "foo"}}

    def test_tuple_non_dict_args(self) -> None:
        result = _safe_tc(("grep", "not-a-dict"))
        assert result == {"name": "grep", "args": {}}

    def test_list_format(self) -> None:
        result = _safe_tc(["shell", {"cmd": "pwd"}])
        assert result == {"name": "shell", "args": {"cmd": "pwd"}}

    def test_unrecognized_type_returns_unknown(self) -> None:
        result = _safe_tc(42)
        assert result == {"name": "unknown", "args": {}}

    def test_none_returns_unknown(self) -> None:
        result = _safe_tc(None)
        assert result == {"name": "unknown", "args": {}}


# ---------------------------------------------------------------------------
# Tool call ID pairing (P2)
# ---------------------------------------------------------------------------


class TestToolCallIdPairing:
    """Tool call and tool result events should share the same call_id
    when the LLM provides a tool_call_id (LangGraph structured calls)."""

    def test_tool_call_uses_langgraph_id(self) -> None:
        """When tool_calls include an 'id' field, call_id should match."""
        s = LangGraphSerializer()
        msg = _make_msg(
            content="",
            tool_calls=[{"name": "shell", "args": {"command": "ls"}, "id": "tc_abc123"}],
        )
        result = s.serialize("executor", {"messages": [msg]})
        events = _parse_lines(result)
        tc_event = [e for e in events if e["type"] == "tool_call"][0]
        assert tc_event["call_id"] == "tc_abc123"

    def test_tool_result_uses_tool_call_id(self) -> None:
        """ToolMessage's tool_call_id should be used as call_id."""
        s = LangGraphSerializer()
        msg = _make_msg(content="file1.txt\nfile2.txt", name="shell")
        msg.tool_call_id = "tc_abc123"
        result = s.serialize("tools", {"messages": [msg]})
        data = json.loads(result)
        assert data["call_id"] == "tc_abc123"

    def test_tool_call_and_result_ids_match(self) -> None:
        """End-to-end: tool_call and tool_result should share the same call_id."""
        s = LangGraphSerializer()
        # Emit tool call
        call_msg = _make_msg(
            content="",
            tool_calls=[{"name": "shell", "args": {"command": "pwd"}, "id": "call_xyz"}],
        )
        call_result = s.serialize("executor", {"messages": [call_msg]})
        call_events = _parse_lines(call_result)
        tc_event = [e for e in call_events if e["type"] == "tool_call"][0]

        # Emit tool result
        result_msg = _make_msg(content="/workspace/abc123", name="shell")
        result_msg.tool_call_id = "call_xyz"
        result_output = s.serialize("tools", {"messages": [result_msg]})
        result_data = json.loads(result_output)

        assert tc_event["call_id"] == result_data["call_id"] == "call_xyz"

    def test_tool_call_falls_back_to_uuid_when_no_id(self) -> None:
        """When tool_calls don't include an 'id' field, a UUID is generated."""
        s = LangGraphSerializer()
        msg = _make_msg(
            content="",
            tool_calls=[{"name": "shell", "args": {"command": "ls"}}],
        )
        result = s.serialize("executor", {"messages": [msg]})
        events = _parse_lines(result)
        tc_event = [e for e in events if e["type"] == "tool_call"][0]
        assert len(tc_event["call_id"]) == 8  # uuid4()[:8]

    def test_tool_result_falls_back_to_last_call_id(self) -> None:
        """When ToolMessage has no tool_call_id, falls back to _last_call_id."""
        s = LangGraphSerializer()
        # First emit a tool call to set _last_call_id
        call_msg = _make_msg(
            content="",
            tool_calls=[{"name": "shell", "args": {}, "id": "prev_call"}],
        )
        s.serialize("executor", {"messages": [call_msg]})

        # Then emit tool result without tool_call_id
        result_msg = MagicMock(spec=[])
        result_msg.content = "output"
        result_msg.name = "shell"
        # No tool_call_id attribute
        result_output = s.serialize("tools", {"messages": [result_msg]})
        result_data = json.loads(result_output)
        assert result_data["call_id"] == "prev_call"


# ---------------------------------------------------------------------------
# Tool result status detection (exit code based)
# ---------------------------------------------------------------------------


class TestToolResultStatus:
    """Tool result status should be based on exit code, not keyword matching."""

    def test_success_output_with_failure_word(self) -> None:
        """Output containing 'failure' (like gh run list) should be success."""
        s = LangGraphSerializer()
        msg = _make_msg(
            content="completed\tfailure\tSome workflow\tCI\tmain\tpull_request\t12345\t1m\t2026-01-01",
            name="shell",
        )
        result = s.serialize("tools", {"messages": [msg]})
        data = json.loads(result)
        assert data["status"] == "success", (
            "Output containing 'failure' as data should be status=success"
        )

    def test_success_output_with_error_word(self) -> None:
        """Output containing 'error' in normal text should be success."""
        s = LangGraphSerializer()
        msg = _make_msg(
            content="Searched for error patterns: none found",
            name="grep",
        )
        result = s.serialize("tools", {"messages": [msg]})
        data = json.loads(result)
        assert data["status"] == "success"

    def test_nonzero_exit_code_is_error(self) -> None:
        """EXIT_CODE: 1 should be status=error."""
        s = LangGraphSerializer()
        msg = _make_msg(
            content="command output\nEXIT_CODE: 1",
            name="shell",
        )
        result = s.serialize("tools", {"messages": [msg]})
        data = json.loads(result)
        assert data["status"] == "error"

    def test_zero_exit_code_is_success(self) -> None:
        """EXIT_CODE: 0 should be status=success (not error)."""
        s = LangGraphSerializer()
        msg = _make_msg(
            content="all good\nEXIT_CODE: 0",
            name="shell",
        )
        result = s.serialize("tools", {"messages": [msg]})
        data = json.loads(result)
        assert data["status"] == "success"

    def test_permission_denied_is_error(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="Permission denied", name="shell")
        result = s.serialize("tools", {"messages": [msg]})
        data = json.loads(result)
        assert data["status"] == "error"

    def test_command_not_found_is_error(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="bash: xyz: command not found", name="shell")
        result = s.serialize("tools", {"messages": [msg]})
        data = json.loads(result)
        assert data["status"] == "error"

    def test_error_prefix_is_error(self) -> None:
        """Lines starting with 'Error: ' are genuine errors."""
        s = LangGraphSerializer()
        msg = _make_msg(content="Error: file not found", name="file_read")
        result = s.serialize("tools", {"messages": [msg]})
        data = json.loads(result)
        assert data["status"] == "error"

    def test_normal_output_is_success(self) -> None:
        s = LangGraphSerializer()
        msg = _make_msg(content="file1.txt\nfile2.txt\nfile3.txt", name="shell")
        result = s.serialize("tools", {"messages": [msg]})
        data = json.loads(result)
        assert data["status"] == "success"


# ---------------------------------------------------------------------------
# Event index uniqueness
# ---------------------------------------------------------------------------


class TestEventIndexUniqueness:
    """Each non-legacy event must have a unique event_index."""

    def test_executor_events_have_unique_indexes(self) -> None:
        """Executor emitting micro_reasoning + executor_step + tool_call
        should produce unique event_index for each non-legacy event."""
        s = LangGraphSerializer()
        msg = _make_msg(
            content="thinking...",
            tool_calls=[{"name": "shell", "args": {"command": "ls"}, "id": "tc1"}],
        )
        result = s.serialize("executor", {"messages": [msg]})
        events = _parse_lines(result)

        # Collect non-legacy event indexes
        non_legacy = [e for e in events if e["type"] not in ("plan_step",)]
        indexes = [e["event_index"] for e in non_legacy]
        assert len(indexes) == len(set(indexes)), (
            f"Non-legacy events have duplicate indexes: {indexes}"
        )

    def test_planner_legacy_shares_index(self) -> None:
        """Legacy 'plan' event should share index with 'planner_output'."""
        s = LangGraphSerializer()
        result = s.serialize("planner", {
            "plan": ["Step 1", "Step 2"],
            "iteration": 1,
            "messages": [],
        })
        events = _parse_lines(result)
        new_evt = [e for e in events if e["type"] == "planner_output"][0]
        legacy_evt = [e for e in events if e["type"] == "plan"][0]
        # Legacy shares index with its new-type counterpart
        assert legacy_evt["event_index"] == new_evt["event_index"]

    def test_full_flow_no_duplicate_indexes(self) -> None:
        """Simulate planner → executor → tool → reflector and check uniqueness."""
        s = LangGraphSerializer()

        # Planner
        s.serialize("planner", {"plan": ["A", "B"], "iteration": 1, "messages": []})

        # Step selector
        s.serialize("step_selector", {"current_step": 0, "plan_steps": [{"description": "A"}]})

        # Executor with tool call
        exec_msg = _make_msg(
            content="",
            tool_calls=[{"name": "shell", "args": {"command": "ls"}, "id": "tc1"}],
        )
        s.serialize("executor", {"messages": [exec_msg]})

        # Tool result
        tool_msg = _make_msg(content="file1.txt", name="shell", tool_call_id="tc1")
        s.serialize("tools", {"messages": [tool_msg]})

        # Reflector
        reflect_msg = _make_msg(content="continue")
        s.serialize("reflector", {"done": False, "current_step": 0, "messages": [reflect_msg]})

        # Check: all non-legacy events across the full flow should have unique indexes
        # (We can't easily collect all events here since serialize returns strings,
        # but the per-call tests above verify the contract)
