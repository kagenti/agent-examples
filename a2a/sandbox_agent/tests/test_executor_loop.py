"""Tests for executor tool loop behavior.

Covers:
  1. No orphaned tool_call/tool_result — every call has a result, every result has a call
  2. Executor does not exit tool loop prematurely on text responses
  3. sub_index continuity between executor and tools nodes
  4. Dedup removal — structured tool calls don't need dedup
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from sandbox_agent.event_serializer import LangGraphSerializer
from sandbox_agent.reasoning import executor_node

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="Do task")],
        "plan": ["Clone repo", "List failures"],
        "current_step": 0,
        "step_results": [],
        "iteration": 1,
        "context_id": "test-ctx",
        "workspace_path": "/workspace/test-ctx",
        "_tool_call_count": 0,
        "_no_tool_count": 0,
    }
    state.update(overrides)
    return state


def _parse_lines(result: str) -> list[dict]:
    return [json.loads(line) for line in result.strip().split("\n") if line.strip()]


def _content_events(result: str) -> list[dict]:
    """Parse lines and filter out meta events (node_transition)."""
    skip = {"node_transition"}
    return [e for e in _parse_lines(result) if e.get("type") not in skip]


# ---------------------------------------------------------------------------
# 1. No dedup — structured tool calls should not be deduped
# ---------------------------------------------------------------------------


class TestNoDedupForStructuredCalls:
    """With tool_choice='any' (structured calls), dedup should not activate."""

    @pytest.mark.asyncio
    async def test_same_tool_different_id_not_deduped(self) -> None:
        """Two shell(ls) calls with different IDs should both execute."""
        llm = AsyncMock()
        llm.ainvoke.return_value = AIMessage(
            content="",
            tool_calls=[{"name": "shell", "args": {"command": "ls"}, "id": "call_2"}],
        )

        state = _base_state(
            _tool_call_count=1,
            messages=[
                HumanMessage(content="Do task"),
                SystemMessage(content="[STEP_BOUNDARY 0] Clone repo"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "shell", "args": {"command": "ls"}, "id": "call_1"}],
                ),
                ToolMessage(content="file1.txt", tool_call_id="call_1", name="shell"),
            ],
        )
        result = await executor_node(state, llm)

        # Should NOT be deduped — the tool call has a different ID
        assert result.get("_dedup") is not True, "Structured tool call with unique ID should not be deduped"
        # Should have the tool call in the response
        resp_msg = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert any(m.tool_calls for m in resp_msg), "Response should contain tool_calls (not deduped)"


# ---------------------------------------------------------------------------
# 2. Executor should not exit tool loop on text between tool calls
# ---------------------------------------------------------------------------


class TestExecutorToolLoopContinuation:
    """Executor should continue tool loop when it has already called tools."""

    @pytest.mark.asyncio
    async def test_text_response_after_tool_calls_is_completion(self) -> None:
        """Text response after tool_call_count > 0 is step completion, not stall."""
        llm = AsyncMock()
        llm.ainvoke.return_value = AIMessage(content="Step completed successfully.")

        state = _base_state(
            _tool_call_count=3,  # Already called 3 tools
            _no_tool_count=0,
        )
        result = await executor_node(state, llm)

        # Should NOT mark as failed — text after tool calls is normal completion
        content = str(result["messages"][-1].content)
        assert "failed" not in content.lower(), f"Text response after tool calls should not be 'failed': {content}"
        # no_tool_count should remain 0 (not incremented when tool_call_count > 0)
        assert result.get("_no_tool_count", 0) == 0

    @pytest.mark.asyncio
    async def test_first_no_tool_response_warns_but_continues(self) -> None:
        """First text response with no prior tool calls warns but doesn't fail."""
        llm = AsyncMock()
        llm.ainvoke.return_value = AIMessage(content="Let me think about this...")

        state = _base_state(
            _tool_call_count=0,  # No tools called yet
            _no_tool_count=0,  # First attempt
        )
        result = await executor_node(state, llm)

        # Should increment no_tool_count but not fail
        assert result.get("_no_tool_count") == 1
        content = str(result["messages"][-1].content)
        assert "failed" not in content.lower()

    @pytest.mark.asyncio
    async def test_second_no_tool_response_fails(self) -> None:
        """Second consecutive text response with no tools → step failed."""
        llm = AsyncMock()
        llm.ainvoke.return_value = AIMessage(content="Still thinking...")

        state = _base_state(
            _tool_call_count=0,
            _no_tool_count=1,  # Already failed once
        )
        result = await executor_node(state, llm)

        content = str(result["messages"][-1].content)
        assert "failed" in content.lower()


# ---------------------------------------------------------------------------
# 3. sub_index continuity between executor and tools nodes
# ---------------------------------------------------------------------------


class TestSubIndexContinuity:
    """Tool result sub_index should continue from executor's last sub_index."""

    def test_tool_result_follows_executor(self) -> None:
        """After executor emits events at sub_index 0-2, tools should be 3."""
        s = LangGraphSerializer()

        # Executor with tool call
        from unittest.mock import MagicMock

        exec_msg = MagicMock(spec=["content", "tool_calls", "name", "tool_call_id"])
        exec_msg.content = ""
        exec_msg.tool_calls = [{"name": "shell", "args": {"command": "ls"}, "id": "tc1"}]
        exec_msg.name = "unknown"
        exec_msg.tool_call_id = None

        exec_result = s.serialize("executor", {"messages": [exec_msg]})
        exec_events = _content_events(exec_result)
        # Get max sub_index from executor events
        exec_max_si = max(e.get("sub_index", 0) for e in exec_events)

        # Tools node
        tool_msg = MagicMock(spec=["content", "tool_calls", "name", "tool_call_id"])
        tool_msg.content = "file1.txt"
        tool_msg.tool_calls = []
        tool_msg.name = "shell"
        tool_msg.tool_call_id = "tc1"

        tool_result = s.serialize("tools", {"messages": [tool_msg]})
        tool_events = _content_events(tool_result)

        tool_si = tool_events[0].get("sub_index")
        assert tool_si == exec_max_si + 1, (
            f"Tool result sub_index ({tool_si}) should be executor max ({exec_max_si}) + 1"
        )

    def test_multiple_executor_tools_cycles(self) -> None:
        """Two executor→tools cycles should have incrementing node_visit."""
        s = LangGraphSerializer()
        from unittest.mock import MagicMock

        visits = []
        for i in range(2):
            exec_msg = MagicMock(spec=["content", "tool_calls", "name", "tool_call_id"])
            exec_msg.content = ""
            exec_msg.tool_calls = [{"name": "shell", "args": {}, "id": f"tc{i}"}]
            exec_msg.name = "unknown"
            exec_msg.tool_call_id = None

            exec_r = s.serialize("executor", {"messages": [exec_msg]})
            exec_events = _content_events(exec_r)
            exec_nv = exec_events[0]["node_visit"]
            visits.append(exec_nv)

            tool_msg = MagicMock(spec=["content", "tool_calls", "name", "tool_call_id"])
            tool_msg.content = f"output{i}"
            tool_msg.tool_calls = []
            tool_msg.name = "shell"
            tool_msg.tool_call_id = f"tc{i}"

            tool_r = s.serialize("tools", {"messages": [tool_msg]})
            tool_events = _content_events(tool_r)
            tool_nv = tool_events[0]["node_visit"]
            # Tools should share executor's node_visit
            assert tool_nv == exec_nv, f"Cycle {i}: tools nv={tool_nv} should match executor nv={exec_nv}"

        # Both executor re-entries in tool loop share the SAME node_visit
        assert visits[0] == visits[1], f"Executor re-entries in tool loop should share visit: {visits}"


# ---------------------------------------------------------------------------
# 4. Event pairing — every tool_call has a tool_result and vice versa
# ---------------------------------------------------------------------------


class TestEventPairing:
    """Serialized events should have matching tool_call/tool_result pairs."""

    def test_executor_plus_tools_produces_pair(self) -> None:
        """executor(tool_call) + tools(tool_result) should share call_id."""
        s = LangGraphSerializer()
        from unittest.mock import MagicMock

        exec_msg = MagicMock(spec=["content", "tool_calls", "name", "tool_call_id"])
        exec_msg.content = ""
        exec_msg.tool_calls = [{"name": "shell", "args": {"command": "pwd"}, "id": "call_xyz"}]
        exec_msg.name = "unknown"
        exec_msg.tool_call_id = None

        exec_r = s.serialize("executor", {"messages": [exec_msg]})
        exec_events = _parse_lines(exec_r)
        tc_events = [e for e in exec_events if e["type"] == "tool_call"]
        assert len(tc_events) == 1
        tc_call_id = tc_events[0]["call_id"]

        tool_msg = MagicMock(spec=["content", "tool_calls", "name", "tool_call_id"])
        tool_msg.content = "/workspace/test"
        tool_msg.tool_calls = []
        tool_msg.name = "shell"
        tool_msg.tool_call_id = "call_xyz"

        tool_r = s.serialize("tools", {"messages": [tool_msg]})
        tool_events = _parse_lines(tool_r)
        tr_events = [e for e in tool_events if e["type"] == "tool_result"]
        assert len(tr_events) == 1
        assert tr_events[0]["call_id"] == tc_call_id

    def test_no_orphaned_tool_results_in_full_flow(self) -> None:
        """Full executor→tools flow should produce no orphans."""
        s = LangGraphSerializer()
        from unittest.mock import MagicMock

        all_events = []

        # 3 executor→tools cycles
        for i in range(3):
            exec_msg = MagicMock(spec=["content", "tool_calls", "name", "tool_call_id"])
            exec_msg.content = f"thinking {i}"
            exec_msg.tool_calls = [{"name": "shell", "args": {"command": f"cmd{i}"}, "id": f"call_{i}"}]
            exec_msg.name = "unknown"
            exec_msg.tool_call_id = None

            r = s.serialize("executor", {"messages": [exec_msg]})
            all_events.extend(_parse_lines(r))

            tool_msg = MagicMock(spec=["content", "tool_calls", "name", "tool_call_id"])
            tool_msg.content = f"output {i}"
            tool_msg.tool_calls = []
            tool_msg.name = "shell"
            tool_msg.tool_call_id = f"call_{i}"

            r = s.serialize("tools", {"messages": [tool_msg]})
            all_events.extend(_parse_lines(r))

        tc_ids = {e["call_id"] for e in all_events if e["type"] == "tool_call"}
        tr_ids = {e["call_id"] for e in all_events if e["type"] == "tool_result"}

        orphan_calls = tc_ids - tr_ids
        orphan_results = tr_ids - tc_ids
        assert not orphan_calls, f"Orphaned tool_calls (no result): {orphan_calls}"
        assert not orphan_results, f"Orphaned tool_results (no call): {orphan_results}"
