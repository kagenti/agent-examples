"""Tests for the node_visit indexing model in the event serializer.

The graph visits nodes in a clear sequence:
  router(1) → planner(2) → step_selector(3) → executor(4) → [tools] →
  executor(5) → [tools] → reflector(6) → step_selector(7) → ...

Each numbered item is a "node visit" — the main section in the UI.
Sub-events within a visit (micro_reasoning, tool_call, tool_result)
share the same node_visit but get their own sub_index.

The "tools" node is special — its tool_result events are associated
with the PRECEDING executor's node_visit (not a separate visit).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from sandbox_agent.event_serializer import LangGraphSerializer


def _make_msg(
    content: str = "",
    tool_calls: list | None = None,
    name: str | None = None,
    tool_call_id: str | None = None,
) -> MagicMock:
    msg = MagicMock(spec=["content", "tool_calls", "name", "tool_call_id"])
    msg.content = content
    msg.tool_calls = tool_calls or []
    msg.name = name if name is not None else "unknown"
    msg.tool_call_id = tool_call_id
    return msg


def _parse_lines(result: str) -> list[dict]:
    return [json.loads(line) for line in result.strip().split("\n") if line.strip()]


def _get_non_legacy(events: list[dict]) -> list[dict]:
    """Filter out legacy event types and meta events that share indexes."""
    skip = {"plan", "plan_step", "reflection", "node_transition"}
    return [e for e in events if e.get("type") not in skip]


# ---------------------------------------------------------------------------
# node_visit: each serialize() call = one node visit
# ---------------------------------------------------------------------------


class TestNodeVisitField:
    """Each serialize() call should produce events with the same node_visit."""

    def test_router_has_node_visit_1(self) -> None:
        s = LangGraphSerializer()
        result = s.serialize("router", {"_route": "plan"})
        events = _parse_lines(result)
        for e in events:
            assert "node_visit" in e, f"Event {e['type']} missing node_visit"
            assert e["node_visit"] == 1

    def test_planner_has_node_visit_2(self) -> None:
        s = LangGraphSerializer()
        # Visit 1: router
        s.serialize("router", {"_route": "plan"})
        # Visit 2: planner
        result = s.serialize(
            "planner",
            {
                "plan": ["Clone repo", "List failures"],
                "iteration": 1,
                "messages": [],
            },
        )
        events = _parse_lines(result)
        non_legacy = _get_non_legacy(events)
        for e in non_legacy:
            assert e["node_visit"] == 2, f"Planner event {e['type']} has visit {e.get('node_visit')}, expected 2"

    def test_tools_node_inherits_executor_visit(self) -> None:
        """Tool results from 'tools' node should share the preceding executor's node_visit."""
        s = LangGraphSerializer()
        # Visit 1: executor with tool call
        exec_msg = _make_msg(
            content="",
            tool_calls=[{"name": "shell", "args": {"command": "ls"}, "id": "tc1"}],
        )
        exec_result = s.serialize("executor", {"messages": [exec_msg]})
        exec_events = _get_non_legacy(_parse_lines(exec_result))
        exec_visit = exec_events[0]["node_visit"]

        # Visit (same): tools node — should inherit executor's visit
        tool_msg = _make_msg(content="file1.txt", name="shell", tool_call_id="tc1")
        tool_result = s.serialize("tools", {"messages": [tool_msg]})
        tool_events = _get_non_legacy(_parse_lines(tool_result))
        for e in tool_events:
            assert e["node_visit"] == exec_visit, (
                f"Tool result should inherit executor visit {exec_visit}, got {e.get('node_visit')}"
            )

    def test_sequential_visits_increment(self) -> None:
        """Full flow: different node types get incrementing visit numbers."""
        s = LangGraphSerializer()
        visits = []

        # router
        r = s.serialize("router", {"_route": "plan"})
        visits.append(_parse_lines(r)[0]["node_visit"])

        # planner
        r = s.serialize("planner", {"plan": ["A"], "iteration": 1, "messages": []})
        visits.append(_get_non_legacy(_parse_lines(r))[0]["node_visit"])

        # step_selector
        r = s.serialize("step_selector", {"current_step": 0, "plan_steps": [{"description": "A"}]})
        visits.append(_get_non_legacy(_parse_lines(r))[0]["node_visit"])

        # executor
        msg = _make_msg(content="", tool_calls=[{"name": "shell", "args": {}, "id": "t1"}])
        r = s.serialize("executor", {"messages": [msg]})
        visits.append(_get_non_legacy(_parse_lines(r))[0]["node_visit"])

        # tools (should NOT increment — inherits executor's visit)
        tool_msg = _make_msg(content="ok", name="shell", tool_call_id="t1")
        r = s.serialize("tools", {"messages": [tool_msg]})
        tool_visit = _get_non_legacy(_parse_lines(r))[0]["node_visit"]

        # executor again (same node type re-entering — stays on SAME visit)
        msg2 = _make_msg(content="", tool_calls=[{"name": "shell", "args": {}, "id": "t2"}])
        r = s.serialize("executor", {"messages": [msg2]})
        exec2_visit = _get_non_legacy(_parse_lines(r))[0]["node_visit"]

        # reflector (different node type — NEW visit)
        ref_msg = _make_msg(content="continue")
        r = s.serialize("reflector", {"done": False, "current_step": 0, "messages": [ref_msg]})
        visits.append(_get_non_legacy(_parse_lines(r))[0]["node_visit"])

        # Visits: router=1, planner=2, step_selector=3, executor=4, reflector=5
        assert visits == [1, 2, 3, 4, 5], f"Visits should be sequential: {visits}"
        # tools inherits executor's visit
        assert tool_visit == 4, f"Tools visit should match executor (4), got {tool_visit}"
        # executor re-entry stays on same visit (tool loop)
        assert exec2_visit == 4, f"Executor re-entry should stay on visit 4, got {exec2_visit}"

    def test_executor_tool_loop_same_visit(self) -> None:
        """Multiple executor→tools→executor cycles share the same node_visit."""
        s = LangGraphSerializer()
        # Simulate: step_selector → executor → tools → executor → tools → executor

        s.serialize("step_selector", {"current_step": 0, "plan_steps": [{"description": "A"}]})

        executor_visits = []
        for i in range(3):
            msg = _make_msg(content="", tool_calls=[{"name": "shell", "args": {}, "id": f"t{i}"}])
            r = s.serialize("executor", {"messages": [msg]})
            executor_visits.append(_get_non_legacy(_parse_lines(r))[0]["node_visit"])

            tool_msg = _make_msg(content=f"out{i}", name="shell", tool_call_id=f"t{i}")
            s.serialize("tools", {"messages": [tool_msg]})

        # All 3 executor calls should share the same node_visit
        assert len(set(executor_visits)) == 1, (
            f"All executor calls in tool loop should share one visit: {executor_visits}"
        )


# ---------------------------------------------------------------------------
# sub_index: position within a node visit
# ---------------------------------------------------------------------------


class TestSubIndex:
    """Events within the same node visit should have sequential sub_index."""

    def test_executor_sub_indexes(self) -> None:
        """Executor emitting micro_reasoning + executor_step + tool_call
        should have sub_index 0, 1, 2 (excluding legacy types)."""
        s = LangGraphSerializer()
        msg = _make_msg(
            content="thinking...",
            tool_calls=[{"name": "shell", "args": {"command": "ls"}, "id": "tc1"}],
        )
        result = s.serialize("executor", {"messages": [msg]})
        events = _get_non_legacy(_parse_lines(result))

        sub_indexes = [e.get("sub_index") for e in events]
        assert sub_indexes == list(range(len(sub_indexes))), f"Sub-indexes should be sequential: {sub_indexes}"

    def test_tool_result_sub_index_continues(self) -> None:
        """Tool result's sub_index should continue from executor's last."""
        s = LangGraphSerializer()
        exec_msg = _make_msg(
            content="",
            tool_calls=[{"name": "shell", "args": {}, "id": "tc1"}],
        )
        exec_result = s.serialize("executor", {"messages": [exec_msg]})
        exec_events = _get_non_legacy(_parse_lines(exec_result))
        last_sub = exec_events[-1].get("sub_index", 0)

        tool_msg = _make_msg(content="output", name="shell", tool_call_id="tc1")
        tool_result = s.serialize("tools", {"messages": [tool_msg]})
        tool_events = _get_non_legacy(_parse_lines(tool_result))
        assert tool_events[0].get("sub_index") == last_sub + 1


# ---------------------------------------------------------------------------
# event_index: global ordering (still needed for total sort)
# ---------------------------------------------------------------------------


class TestGlobalEventIndex:
    """event_index should be globally unique and monotonically increasing."""

    def test_no_duplicate_event_index_in_flow(self) -> None:
        """Full flow should produce unique event_index across all events."""
        s = LangGraphSerializer()
        all_events = []

        s.serialize("router", {"_route": "plan"})
        r = s.serialize("planner", {"plan": ["A", "B"], "iteration": 1, "messages": []})
        all_events.extend(_parse_lines(r))

        r = s.serialize("step_selector", {"current_step": 0, "plan_steps": [{"description": "A"}]})
        all_events.extend(_parse_lines(r))

        msg = _make_msg(content="", tool_calls=[{"name": "shell", "args": {}, "id": "t1"}])
        r = s.serialize("executor", {"messages": [msg]})
        all_events.extend(_parse_lines(r))

        tool_msg = _make_msg(content="ok", name="shell", tool_call_id="t1")
        r = s.serialize("tools", {"messages": [tool_msg]})
        all_events.extend(_parse_lines(r))

        ref_msg = _make_msg(content="continue")
        r = s.serialize("reflector", {"done": False, "current_step": 0, "messages": [ref_msg]})
        all_events.extend(_parse_lines(r))

        non_legacy = _get_non_legacy(all_events)
        indexes = [e["event_index"] for e in non_legacy]
        assert len(indexes) == len(set(indexes)), f"Duplicate event_index values: {indexes}"
        # Should be monotonically increasing
        for i in range(1, len(indexes)):
            assert indexes[i] > indexes[i - 1], f"event_index not monotonic at position {i}: {indexes}"


# ---------------------------------------------------------------------------
# Micro-reasoning counter resets per step
# ---------------------------------------------------------------------------


class TestMicroStepCounter:
    """Micro-reasoning sub_index should reset on step transitions."""

    def test_micro_step_resets_on_step_selector(self) -> None:
        """After step_selector, micro_step should restart from 0."""
        s = LangGraphSerializer()

        # Step 1: executor with some micro-reasoning
        msg1 = _make_msg(content="thinking", tool_calls=[{"name": "shell", "args": {}, "id": "t1"}])
        s.serialize("executor", {"messages": [msg1], "current_step": 0})
        s.serialize("tools", {"messages": [_make_msg(content="ok", name="shell", tool_call_id="t1")]})

        # Another executor call (continuing step 1)
        msg2 = _make_msg(content="more thinking", tool_calls=[{"name": "shell", "args": {}, "id": "t2"}])
        r2 = s.serialize("executor", {"messages": [msg2], "current_step": 0})
        events2 = _parse_lines(r2)
        micro2 = [e for e in events2 if e["type"] == "micro_reasoning"]
        if micro2:
            micro_before = micro2[0].get("micro_step", 0) # noqa: F841

        # Reflector + step_selector transition
        s.serialize("reflector", {"done": False, "current_step": 0, "messages": [_make_msg(content="continue")]})
        s.serialize("step_selector", {"current_step": 1, "plan_steps": [{"description": "A"}, {"description": "B"}]})

        # Step 2: executor should have micro_step reset
        msg3 = _make_msg(content="new step", tool_calls=[{"name": "shell", "args": {}, "id": "t3"}])
        r3 = s.serialize("executor", {"messages": [msg3], "current_step": 1})
        events3 = _parse_lines(r3)
        micro3 = [e for e in events3 if e["type"] == "micro_reasoning"]
        if micro3:
            assert micro3[0]["micro_step"] == 1, (
                f"micro_step should reset to 1 after step transition, got {micro3[0]['micro_step']}"
            )


# ---------------------------------------------------------------------------
# Plan step field (which plan step is being executed)
# ---------------------------------------------------------------------------


class TestPlanStepField:
    """Each event should have the correct plan step number."""

    def test_step_field_matches_current_step(self) -> None:
        """Events should reflect the actual plan step being executed."""
        s = LangGraphSerializer()

        # Step selector sets current_step=0
        r1 = s.serialize("step_selector", {"current_step": 0, "plan_steps": [{"description": "A"}]})
        e1 = _get_non_legacy(_parse_lines(r1))[0]
        assert e1["step"] == 1, f"Step should be 1 (0-based + 1), got {e1['step']}"

        # Executor for step 0
        msg = _make_msg(content="working")
        r2 = s.serialize("executor", {"messages": [msg], "current_step": 0})
        events2 = _get_non_legacy(_parse_lines(r2))
        for e in events2:
            assert e["step"] == 1, f"Executor event should show step 1, got {e['step']}"

        # Step selector advances to step 1
        r3 = s.serialize(
            "step_selector", {"current_step": 1, "plan_steps": [{"description": "A"}, {"description": "B"}]}
        )
        e3 = _get_non_legacy(_parse_lines(r3))[0]
        assert e3["step"] == 2, f"Step should be 2 after advancing, got {e3['step']}"
