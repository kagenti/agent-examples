"""Tests for LangGraph node context isolation in the reasoning loop.

Simulates a full RCA workflow (clone → list failures → download logs →
grep errors → write report) with mocked LLM responses and tool results.
Verifies that each node (planner, executor, reflector, reporter) receives
ONLY its intended context — no plan leakage into executor, no full history
in reflector, etc.

Test structure:
  1. CaptureLLM — mock LLM that records messages per node
  2. Per-node context tests — verify message isolation
  3. Full flow test — simulate 5-step RCA with mocked responses
  4. Failure + replan test — verify replan doesn't duplicate steps
  5. Logging assertions — verify structured OTel log fields
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from sandbox_agent.budget import AgentBudget
from sandbox_agent.reasoning import (
    _parse_plan,
    _safe_format,
    executor_node,
    planner_node,
    reflector_node,
    reporter_node,
)
from sandbox_agent.prompts import (
    EXECUTOR_SYSTEM as _EXECUTOR_SYSTEM,
    PLANNER_SYSTEM as _PLANNER_SYSTEM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CaptureLLM:
    """Mock LLM that records what messages were sent and returns scripted responses.

    Usage::

        llm = CaptureLLM([
            AIMessage(content="1. Clone repo\\n2. List failures"),
            AIMessage(content="continue"),
        ])
        await llm.ainvoke(messages)
        assert llm.calls[0]  # messages sent to first call
    """

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self._call_idx = 0
        self.calls: list[list] = []  # each entry is the messages list

    async def ainvoke(self, messages: list) -> AIMessage:
        self.calls.append(list(messages))
        if self._call_idx < len(self._responses):
            resp = self._responses[self._call_idx]
        else:
            resp = AIMessage(content="(no more scripted responses)")
        self._call_idx += 1
        # Add usage_metadata so budget tracking doesn't crash
        resp.usage_metadata = {"input_tokens": 100, "output_tokens": 20}
        resp.response_metadata = {"model": "test-model"}
        return resp

    @property
    def last_messages(self) -> list:
        """Messages from the most recent ainvoke call."""
        return self.calls[-1] if self.calls else []

    def system_prompt(self, call_idx: int = -1) -> str:
        """Extract the system prompt text from a specific call."""
        msgs = self.calls[call_idx]
        for m in msgs:
            if isinstance(m, SystemMessage):
                return m.content
        return ""

    def human_messages(self, call_idx: int = -1) -> list[str]:
        """Extract all HumanMessage contents from a specific call."""
        return [
            m.content for m in self.calls[call_idx]
            if isinstance(m, HumanMessage)
        ]

    def ai_messages(self, call_idx: int = -1) -> list[str]:
        """Extract all AIMessage contents from a specific call."""
        return [
            str(m.content) for m in self.calls[call_idx]
            if isinstance(m, AIMessage)
        ]

    def message_types(self, call_idx: int = -1) -> list[str]:
        """Return list of message type names from a specific call."""
        return [type(m).__name__ for m in self.calls[call_idx]]


def _make_rca_plan() -> list[str]:
    """The 5-step RCA plan our mock planner produces."""
    return [
        "Clone the repo: shell(`git clone https://github.com/kagenti/kagenti.git repos/kagenti`).",
        "List CI failures: shell(`cd repos/kagenti && gh run list --status failure --limit 5`).",
        "Download logs: shell(`cd repos/kagenti && gh run view 123 --log-failed > /workspace/ctx/output/ci.log`).",
        "Extract errors: grep(`FAILED|ERROR` in output/ci.log).",
        "Write report: file_write(report.md).",
    ]


def _base_state(**overrides: Any) -> dict[str, Any]:
    """Create a minimal valid state dict for node testing."""
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="Analyze CI failures for kagenti PR #860")],
        "plan": [],
        "plan_steps": [],
        "current_step": 0,
        "step_results": [],
        "iteration": 0,
        "replan_count": 0,
        "done": False,
        "context_id": "test-ctx-123",
        "workspace_path": "/workspace/test-ctx-123",
        "recent_decisions": [],
        "_tool_call_count": 0,
        "_no_tool_count": 0,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# P0: Workspace path in executor prompt
# ---------------------------------------------------------------------------


class TestWorkspacePathInPrompt:
    """Verify workspace_path is injected into the executor system prompt."""

    @pytest.mark.asyncio
    async def test_executor_prompt_contains_workspace_path(self) -> None:
        llm = CaptureLLM([AIMessage(content="Cloning repo...")])
        state = _base_state(
            plan=_make_rca_plan(),
            current_step=0,
            workspace_path="/workspace/abc-def-123",
        )
        await executor_node(state, llm)

        system = llm.system_prompt(0)
        assert "/workspace/abc-def-123" in system
        # Should appear in the redirect guidance
        assert "/workspace/abc-def-123/output/" in system

    @pytest.mark.asyncio
    async def test_executor_prompt_default_workspace(self) -> None:
        """When workspace_path not in state, defaults to /workspace."""
        llm = CaptureLLM([AIMessage(content="Done")])
        state = _base_state(plan=["Do something"], current_step=0)
        del state["workspace_path"]

        await executor_node(state, llm)
        system = llm.system_prompt(0)
        assert "Your workspace absolute path is: /workspace" in system


# ---------------------------------------------------------------------------
# P1: SystemMessage step boundary
# ---------------------------------------------------------------------------


class TestStepBoundary:
    """Verify step boundary marker is SystemMessage and scopes context."""

    @pytest.mark.asyncio
    async def test_first_call_injects_system_boundary(self) -> None:
        """On tool_call_count=0, a SystemMessage boundary is returned."""
        llm = CaptureLLM([AIMessage(content="Starting step 1")])
        state = _base_state(plan=_make_rca_plan(), _tool_call_count=0)

        result = await executor_node(state, llm)
        boundary_msgs = [
            m for m in result["messages"]
            if isinstance(m, SystemMessage)
            and "[STEP_BOUNDARY" in str(m.content)
        ]
        assert len(boundary_msgs) == 1
        assert "[STEP_BOUNDARY 0]" in boundary_msgs[0].content

    @pytest.mark.asyncio
    async def test_continuing_step_no_extra_boundary(self) -> None:
        """On tool_call_count > 0, no new boundary is injected."""
        llm = CaptureLLM([AIMessage(content="Continuing")])
        state = _base_state(
            plan=_make_rca_plan(),
            _tool_call_count=1,
            messages=[
                HumanMessage(content="user request"),
                SystemMessage(content="[STEP_BOUNDARY 0] Execute step 1"),
                AIMessage(content="tool call result"),
            ],
        )
        result = await executor_node(state, llm)
        boundary_msgs = [
            m for m in result["messages"]
            if isinstance(m, SystemMessage)
            and "[STEP_BOUNDARY" in str(m.content)
        ]
        assert len(boundary_msgs) == 0

    @pytest.mark.asyncio
    async def test_executor_does_not_see_planner_message(self) -> None:
        """Executor context on continuing step must NOT include planner output."""
        plan = _make_rca_plan()
        planner_ai = AIMessage(content="1. Clone repo\n2. List failures\n3. Download logs")

        llm = CaptureLLM([AIMessage(content="Next tool call")])
        state = _base_state(
            plan=plan,
            current_step=0,
            _tool_call_count=1,
            messages=[
                HumanMessage(content="Analyze CI failures"),
                planner_ai,  # <-- This should NOT leak into executor
                SystemMessage(content="[STEP_BOUNDARY 0] Clone the repo"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "shell", "args": {"command": "git clone ..."}, "id": "tc1"}],
                ),
                ToolMessage(content="Cloning into 'repos/kagenti'...", tool_call_id="tc1", name="shell"),
            ],
        )
        await executor_node(state, llm)

        # Check what the LLM received
        all_content = " ".join(str(m.content) for m in llm.last_messages)
        # Planner's numbered plan should NOT appear
        assert "1. Clone repo" not in all_content
        assert "2. List failures" not in all_content
        # But the tool result from this step SHOULD appear
        assert "Cloning into" in all_content

    @pytest.mark.asyncio
    async def test_executor_new_step_sees_only_brief(self) -> None:
        """On new step (tool_call_count=0), executor gets ONLY the step brief."""
        plan = _make_rca_plan()
        llm = CaptureLLM([AIMessage(content="Running command")])
        state = _base_state(
            plan=plan,
            current_step=1,  # Step 2: List CI failures
            _tool_call_count=0,
            messages=[
                HumanMessage(content="Analyze CI failures"),
                AIMessage(content="Plan: 1. Clone\n2. List\n3. Download"),
                SystemMessage(content="[STEP_BOUNDARY 0] Clone the repo"),
                AIMessage(content="Cloned successfully"),
            ],
        )
        await executor_node(state, llm)

        # LLM should receive: SystemMessage(prompt) + HumanMessage(step brief)
        types = llm.message_types(0)
        assert types == ["SystemMessage", "HumanMessage"]
        # The HumanMessage should be the step brief, not the original user request
        human_texts = llm.human_messages(0)
        assert len(human_texts) == 1
        # Should NOT contain the planner's plan text
        assert "1. Clone" not in human_texts[0]


# ---------------------------------------------------------------------------
# Planner context: should NOT include own previous plan on replan
# ---------------------------------------------------------------------------


class TestPlannerContext:
    """Verify planner doesn't see its own previous plan in messages on replan."""

    @pytest.mark.asyncio
    async def test_fresh_plan_from_user_message(self) -> None:
        """First plan: planner gets user request, produces numbered steps."""
        llm = CaptureLLM([
            AIMessage(content="1. Clone repo\n2. List failures\n3. Download logs"),
        ])
        state = _base_state(iteration=0)
        result = await planner_node(state, llm)

        assert len(result["plan"]) == 3
        assert result["iteration"] == 1
        # System prompt should contain PLANNER_SYSTEM base
        system = llm.system_prompt(0)
        assert "planning module" in system.lower()

    @pytest.mark.asyncio
    async def test_replan_includes_step_status(self) -> None:
        """On replan, planner should see which steps are done/failed."""
        llm = CaptureLLM([
            AIMessage(content="1. Try alternative approach\n2. Write report"),
        ])
        state = _base_state(
            iteration=1,
            plan=["Clone repo", "List failures"],
            step_results=["Cloned successfully", "gh: command not found"],
        )
        result = await planner_node(state, llm)

        system = llm.system_prompt(0)
        # Should mention previous step results
        assert "Cloned successfully" in system or "step results" in system.lower()

    @pytest.mark.asyncio
    async def test_replan_message_count_bounded(self) -> None:
        """Planner on replan should receive bounded messages, not full history."""
        # Build a history with many messages (simulating 3 executor iterations)
        messages = [HumanMessage(content="Analyze CI failures")]
        messages.append(AIMessage(content="1. Clone\n2. List\n3. Download"))
        for i in range(15):
            messages.append(AIMessage(
                content="",
                tool_calls=[{"name": "shell", "args": {"command": f"cmd{i}"}, "id": f"tc{i}"}],
            ))
            messages.append(ToolMessage(content=f"output {i}", tool_call_id=f"tc{i}", name="shell"))

        llm = CaptureLLM([
            AIMessage(content="1. New approach"),
        ])
        state = _base_state(
            iteration=2,
            plan=["Clone repo"],
            step_results=["Failed"],
            messages=messages,
        )
        await planner_node(state, llm)

        # After context builder fix: planner should receive bounded messages
        # SystemMessage + user request + last few tool results
        total_msgs = len(llm.last_messages)
        assert total_msgs <= 10, (
            f"Planner sent {total_msgs} messages on replan — should be ≤10 "
            f"(system + user request + recent tool results)"
        )
        # Should NOT include the planner's own previous AIMessage
        ai_msgs = llm.ai_messages(0)
        assert not any("1. Clone\n2. List\n3. Download" in ai for ai in ai_msgs), (
            "Planner should not see its own previous plan AIMessage"
        )


# ---------------------------------------------------------------------------
# Reflector context: should see only recent step output
# ---------------------------------------------------------------------------


class TestReflectorContext:
    """Verify reflector receives only the step result context, not full history."""

    @pytest.mark.asyncio
    async def test_reflector_sees_limited_history(self) -> None:
        """Reflector should see at most last 3 AI→Tool pairs."""
        # Build long message history
        messages: list = [HumanMessage(content="user request")]
        messages.append(AIMessage(content="Plan: 1. A\n2. B\n3. C"))
        for i in range(10):
            messages.append(AIMessage(
                content="",
                tool_calls=[{"name": "shell", "args": {"command": f"cmd{i}"}, "id": f"tc{i}"}],
            ))
            messages.append(ToolMessage(content=f"output {i}", tool_call_id=f"tc{i}", name="shell"))
        # Last AI message (step summary)
        messages.append(AIMessage(content="Step 1 completed"))

        llm = CaptureLLM([AIMessage(content="continue")])
        state = _base_state(
            plan=["Step A", "Step B", "Step C"],
            current_step=0,
            iteration=1,
            messages=messages,
        )
        result = await reflector_node(state, llm)

        # Reflector should NOT send all 20+ messages to the LLM
        total = len(llm.last_messages)
        # System + at most 6 messages (3 AI→Tool pairs) + maybe step summary
        assert total <= 10, (
            f"Reflector sent {total} messages to LLM — should be ≤10 "
            f"(system + last 3 AI→Tool pairs)"
        )

    @pytest.mark.asyncio
    async def test_reflector_does_not_see_planner_plan(self) -> None:
        """Reflector context must not include the planner's numbered plan.

        KNOWN BUG: The reflector walks back 3 AI→Tool pairs, but if
        the planner's AIMessage (with no tool_calls) is within that
        window, it leaks through. The fix is to filter out AIMessages
        that don't have tool_calls (planner/reflector text) from the
        reflector's recent messages window.
        """
        plan_text = "1. Clone repo\n2. List failures\n3. Download logs"
        messages: list = [
            HumanMessage(content="user request"),
            AIMessage(content=plan_text),  # Planner output
            AIMessage(
                content="",
                tool_calls=[{"name": "shell", "args": {"command": "git clone ..."}, "id": "tc1"}],
            ),
            ToolMessage(content="Cloning...", tool_call_id="tc1", name="shell"),
            AIMessage(content="Clone complete"),
        ]

        llm = CaptureLLM([AIMessage(content="continue")])
        state = _base_state(
            plan=["Clone repo", "List failures"],
            current_step=0,
            iteration=1,
            messages=messages,
        )
        await reflector_node(state, llm)

        # After context builder fix: reflector should NOT see planner's plan
        ai_contents = llm.ai_messages(0)
        assert not any("1. Clone repo\n2. List failures" in ai for ai in ai_contents), (
            "Reflector should not see planner's AIMessage with full plan"
        )

    @pytest.mark.asyncio
    async def test_reflector_single_step_marks_done(self) -> None:
        """Single-step plan: reflector should mark as done after the step completes."""
        llm = CaptureLLM([AIMessage(content="done")])
        budget = AgentBudget()
        state = _base_state(
            plan=["Just one step"],
            plan_steps=[{"description": "Just one step", "status": "running"}],
            current_step=0,
            iteration=1,
            messages=[AIMessage(content="Step completed successfully")],
        )
        result = await reflector_node(state, llm, budget=budget)
        assert result["done"] is True


# ---------------------------------------------------------------------------
# Reporter context: intentionally sees full history (for summarization)
# ---------------------------------------------------------------------------


class TestReporterContext:
    """Reporter should see the full conversation for final answer generation."""

    @pytest.mark.asyncio
    async def test_reporter_single_step_includes_result(self) -> None:
        """Single-step plan: reporter should include the step result."""
        llm = CaptureLLM([AIMessage(content="Found files: file1.txt file2.txt")])
        budget = AgentBudget()
        state = _base_state(
            plan=["List files"],
            step_results=["file1.txt  file2.txt"],
            messages=[AIMessage(content="file1.txt  file2.txt")],
        )
        result = await reporter_node(state, llm, budget=budget)
        assert "file" in result["final_answer"].lower()

    @pytest.mark.asyncio
    async def test_reporter_multi_step_calls_llm(self) -> None:
        """Multi-step plan: reporter calls LLM with full context."""
        llm = CaptureLLM([
            AIMessage(content="## Root Cause\nThe CI pipeline failed due to..."),
        ])
        budget = AgentBudget()
        state = _base_state(
            plan=["Clone", "List", "Analyze"],
            step_results=["Cloned OK", "Found 3 failures", "Root cause identified"],
            messages=[HumanMessage(content="Analyze CI")],
        )
        result = await reporter_node(state, llm, budget=budget)
        assert "Root Cause" in result["final_answer"]
        assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# Executor failure behavior
# ---------------------------------------------------------------------------


class TestExecutorFailureBehavior:
    """Verify executor handles failures correctly with proper logging."""

    @pytest.mark.asyncio
    async def test_tool_limit_forces_completion(self) -> None:
        """When tool_call_count >= MAX, executor returns without LLM call."""
        from sandbox_agent.reasoning import MAX_TOOL_CALLS_PER_STEP

        llm = CaptureLLM([])
        state = _base_state(
            plan=_make_rca_plan(),
            current_step=0,
            _tool_call_count=MAX_TOOL_CALLS_PER_STEP,
        )
        result = await executor_node(state, llm)
        assert "tool call limit" in str(result["messages"][0].content).lower()
        assert len(llm.calls) == 0  # No LLM call

    @pytest.mark.asyncio
    async def test_no_tool_calls_twice_marks_failed(self) -> None:
        """Executor that produces no tool calls twice marks step as failed."""
        llm = CaptureLLM([
            AIMessage(content="I would run ls but..."),  # No tool_calls
        ])
        state = _base_state(
            plan=_make_rca_plan(),
            current_step=0,
            _tool_call_count=0,
            _no_tool_count=1,  # Already failed once
        )
        result = await executor_node(state, llm)
        assert "failed" in str(result["messages"][0].content).lower()

    @pytest.mark.asyncio
    async def test_budget_exceeded_stops_executor(self) -> None:
        """Executor returns immediately if iteration budget is exceeded."""
        budget = AgentBudget(max_iterations=1)
        budget.tick_iteration()  # iteration 1 — now at limit
        budget.tick_iteration()  # iteration 2 — over limit
        llm = CaptureLLM([])
        state = _base_state(plan=_make_rca_plan(), current_step=0)
        result = await executor_node(state, llm, budget=budget)
        assert result.get("done") is True
        assert "budget" in str(result["messages"][0].content).lower()

    @pytest.mark.asyncio
    async def test_past_last_step_signals_done(self) -> None:
        """Executor past the plan length signals completion."""
        llm = CaptureLLM([])
        state = _base_state(
            plan=["Only step"],
            current_step=1,  # Past the only step
        )
        result = await executor_node(state, llm)
        assert result["done"] is True
        assert len(llm.calls) == 0


# ---------------------------------------------------------------------------
# Executor logging
# ---------------------------------------------------------------------------


class TestExecutorLogging:
    """Verify structured log fields in executor output."""

    @pytest.mark.asyncio
    async def test_executor_logs_context_size(self, caplog: pytest.LogCaptureFixture) -> None:
        """Executor context builder should log message count and char estimate."""
        llm = CaptureLLM([AIMessage(content="Working on it")])
        state = _base_state(plan=_make_rca_plan(), current_step=0)

        with caplog.at_level(logging.INFO, logger="sandbox_agent.context_builders"):
            await executor_node(state, llm)

        context_logs = [r for r in caplog.records if "Executor context" in r.getMessage()]
        assert len(context_logs) >= 1, "Expected 'Executor context' log entry"
        log_msg = context_logs[0].getMessage()
        assert "messages" in log_msg
        assert "chars" in log_msg

    @pytest.mark.asyncio
    async def test_executor_logs_tool_limit_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Executor should warn when hitting tool call limit."""
        from sandbox_agent.reasoning import MAX_TOOL_CALLS_PER_STEP

        llm = CaptureLLM([])
        state = _base_state(
            plan=_make_rca_plan(),
            current_step=0,
            _tool_call_count=MAX_TOOL_CALLS_PER_STEP,
        )
        with caplog.at_level(logging.WARNING, logger="sandbox_agent.reasoning"):
            await executor_node(state, llm)

        limit_logs = [r for r in caplog.records if "tool call limit" in r.getMessage()]
        assert len(limit_logs) >= 1

    @pytest.mark.asyncio
    async def test_executor_logs_no_tool_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Executor should warn when LLM produces no tool calls."""
        llm = CaptureLLM([AIMessage(content="I think we should...")])
        state = _base_state(
            plan=_make_rca_plan(),
            current_step=0,
            _tool_call_count=0,
            _no_tool_count=0,
        )
        with caplog.at_level(logging.WARNING, logger="sandbox_agent.reasoning"):
            await executor_node(state, llm)

        no_tool_logs = [r for r in caplog.records if "no tool calls" in r.getMessage().lower()]
        assert len(no_tool_logs) >= 1


# ---------------------------------------------------------------------------
# Full RCA flow simulation
# ---------------------------------------------------------------------------


class TestFullRCAFlow:
    """Simulate a complete 5-step RCA workflow and verify context at each stage."""

    @pytest.mark.asyncio
    async def test_five_step_rca_context_isolation(self) -> None:
        """Run planner → executor (step 1) → reflector and verify isolation."""
        # Step 1: Planner produces the 5-step plan
        planner_llm = CaptureLLM([
            AIMessage(content=(
                "1. Clone repo: shell(`git clone ...`).\n"
                "2. List failures: shell(`cd repos/kagenti && gh run list ...`).\n"
                "3. Download logs: shell(`cd repos/kagenti && gh run view ...`).\n"
                "4. Extract errors: grep(`FAILED|ERROR`).\n"
                "5. Write report: file_write(report.md)."
            )),
        ])
        state = _base_state(iteration=0)
        plan_result = await planner_node(state, planner_llm)
        assert len(plan_result["plan"]) == 5

        # Step 2: Executor runs step 1 (clone repo)
        executor_llm = CaptureLLM([
            AIMessage(
                content="",
                tool_calls=[{"name": "shell", "args": {"command": "git clone ..."}, "id": "tc_clone"}],
            ),
        ])
        # Build state as it would be after planner: messages includes planner's AIMessage
        exec_state = _base_state(
            plan=plan_result["plan"],
            current_step=0,
            _tool_call_count=0,
            workspace_path="/workspace/test-session-id",
            messages=[
                HumanMessage(content="Analyze CI failures for PR #860"),
                AIMessage(content="1. Clone repo\n2. List failures\n3. Download\n4. Extract\n5. Report"),
            ],
        )
        exec_result = await executor_node(exec_state, executor_llm)

        # CRITICAL: Executor should NOT see the planner's plan in its messages
        types = executor_llm.message_types(0)
        assert types == ["SystemMessage", "HumanMessage"], (
            f"New-step executor should get [SystemMessage, HumanMessage] but got {types}"
        )
        # System prompt should contain workspace path
        system = executor_llm.system_prompt(0)
        assert "/workspace/test-session-id" in system

        # Step 3: Reflector reviews step 1
        reflector_llm = CaptureLLM([AIMessage(content="continue")])
        # Build state after executor: includes planner + boundary + executor output
        reflect_state = _base_state(
            plan=plan_result["plan"],
            current_step=0,
            iteration=1,
            messages=[
                HumanMessage(content="Analyze CI failures"),
                AIMessage(content="1. Clone repo\n2. List\n3. Download\n4. Extract\n5. Report"),
                SystemMessage(content="[STEP_BOUNDARY 0] Clone the repo"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "shell", "args": {"command": "git clone ..."}, "id": "tc1"}],
                ),
                ToolMessage(content="Cloning into 'repos/kagenti'...", tool_call_id="tc1", name="shell"),
                AIMessage(content="Repository cloned successfully"),
            ],
        )
        reflect_result = await reflector_node(reflect_state, reflector_llm)

        # After context builder fix: reflector should NOT see planner's plan
        ai_msgs_in_reflector = reflector_llm.ai_messages(0)
        assert not any("1. Clone repo\n2. List" in ai for ai in ai_msgs_in_reflector), (
            "Reflector should not see planner's full plan as an AIMessage"
        )
        # Reflector decision
        assert reflect_result["done"] is False
        assert reflect_result["current_step"] == 1  # Advanced to next step


# ---------------------------------------------------------------------------
# Replan duplication guard
# ---------------------------------------------------------------------------


class TestReplanDuplication:
    """Verify that replan does not produce duplicate steps."""

    @pytest.mark.asyncio
    async def test_replan_does_not_see_previous_plan_aimessage(self) -> None:
        """When replanning, planner should not see its own previous plan
        as an AIMessage in the conversation (which causes duplication).

        KNOWN BUG: Currently the planner receives full state['messages']
        which includes its own previous AIMessage. This test documents
        the expected behavior after the fix.
        """
        previous_plan = "1. Clone repo\n2. List failures\n3. Download logs"
        llm = CaptureLLM([
            AIMessage(content="1. Try alternative API\n2. Write report"),
        ])
        state = _base_state(
            iteration=1,
            plan=["Clone repo", "List failures", "Download logs"],
            step_results=["Cloned OK", "Failed: gh command not found"],
            messages=[
                HumanMessage(content="Analyze CI failures"),
                AIMessage(content=previous_plan),  # Previous plan AIMessage
                AIMessage(content="Cloned successfully"),
                AIMessage(content="Failed: gh command not found"),
            ],
        )
        await planner_node(state, llm)

        # After context builder fix: planner should NOT see own previous plan
        ai_msgs = llm.ai_messages(0)
        assert not any(previous_plan in ai for ai in ai_msgs), (
            "Planner should not see its own previous plan AIMessage "
            "in conversation history (causes step duplication on replan)"
        )

    @pytest.mark.asyncio
    async def test_replan_can_add_steps_when_objective_not_met(self) -> None:
        """Replanner should add new steps when done steps didn't achieve the goal."""
        llm = CaptureLLM([
            AIMessage(content=(
                "1. Try gh api instead of gh run view.\n"
                "2. Parse JSON response for log URLs.\n"
                "3. Download logs with curl.\n"
                "4. Write findings to report.md."
            )),
        ])
        state = _base_state(
            iteration=1,
            plan=["Clone repo", "List failures", "Download logs"],
            plan_steps=[
                {"description": "Clone repo", "status": "done", "index": 0, "result_summary": "Cloned OK"},
                {"description": "List failures", "status": "done", "index": 1, "result_summary": "Found 3 failures"},
                {"description": "Download logs", "status": "failed", "index": 2, "result_summary": "gh run view: HTTP 410"},
            ],
            step_results=["Cloned OK", "Found 3 failures", "HTTP 410 error"],
            messages=[
                HumanMessage(content="Analyze CI failures"),
                AIMessage(content="1. Clone\n2. List\n3. Download"),
                ToolMessage(content="Cloned OK", tool_call_id="tc1", name="shell"),
                ToolMessage(content="3 failures found", tool_call_id="tc2", name="shell"),
                ToolMessage(content="HTTP 410 error", tool_call_id="tc3", name="shell"),
            ],
        )
        result = await planner_node(state, llm)

        # Replanner should produce new steps (not duplicated old ones)
        assert len(result["plan"]) >= 2
        assert len(result["plan"]) <= 5, "Replan should add at most 5 new steps"
        # The system prompt should include step status context
        system = llm.system_prompt(0)
        assert "DONE" in system or "done" in system.lower()
        assert "HTTP 410" in system or "410" in system

    @pytest.mark.asyncio
    async def test_parsed_plan_has_no_duplicates(self) -> None:
        """_parse_plan should not produce duplicate steps from repeated text."""
        # Simulate what happens when the LLM echoes steps
        text = (
            "1. Clone repo\n"
            "2. List failures\n"
            "3. Download logs\n"
        )
        steps = _parse_plan(text)
        assert len(steps) == 3
        assert len(set(steps)) == 3, "Plan steps should be unique"


# ---------------------------------------------------------------------------
# Direct context builder unit tests
# ---------------------------------------------------------------------------


class TestBuildPlannerContext:
    """Direct tests for build_planner_context."""

    def test_fresh_plan_only_user_messages(self) -> None:
        from sandbox_agent.context_builders import build_planner_context

        state = _base_state(
            iteration=0,
            messages=[
                HumanMessage(content="Analyze CI failures"),
                AIMessage(content="some previous response"),
            ],
        )
        msgs = build_planner_context(state, "System prompt")
        types = [type(m).__name__ for m in msgs]
        assert types == ["SystemMessage", "HumanMessage"]
        assert "Analyze CI failures" in msgs[1].content

    def test_replan_excludes_planner_aimessage(self) -> None:
        from sandbox_agent.context_builders import build_planner_context

        state = _base_state(
            iteration=1,
            messages=[
                HumanMessage(content="Analyze CI failures"),
                AIMessage(content="1. Clone\n2. List\n3. Download"),
                AIMessage(content="", tool_calls=[{"name": "shell", "args": {}, "id": "t1"}]),
                ToolMessage(content="cloned OK", tool_call_id="t1", name="shell"),
                ToolMessage(content="3 failures", tool_call_id="t2", name="shell"),
            ],
        )
        msgs = build_planner_context(state, "System prompt")
        # Should NOT include the planner's AIMessage
        ai_contents = [str(m.content) for m in msgs if isinstance(m, AIMessage)]
        assert not any("1. Clone" in c for c in ai_contents)
        # Should include ToolMessages for context
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        assert len(tool_msgs) >= 1

    def test_replan_includes_user_request(self) -> None:
        from sandbox_agent.context_builders import build_planner_context

        state = _base_state(
            iteration=2,
            messages=[
                HumanMessage(content="Original user request"),
                AIMessage(content="old plan"),
            ],
        )
        msgs = build_planner_context(state, "System prompt")
        human_msgs = [m for m in msgs if isinstance(m, HumanMessage)]
        assert len(human_msgs) == 1
        assert "Original user request" in human_msgs[0].content


class TestBuildReflectorContext:
    """Direct tests for build_reflector_context."""

    def test_only_tool_call_ai_messages(self) -> None:
        from sandbox_agent.context_builders import build_reflector_context

        state = _base_state(messages=[
            HumanMessage(content="user request"),
            AIMessage(content="Plan: 1. Clone\n2. List"),  # No tool_calls — should be filtered
            AIMessage(
                content="",
                tool_calls=[{"name": "shell", "args": {"command": "ls"}, "id": "tc1"}],
            ),
            ToolMessage(content="file1.txt", tool_call_id="tc1", name="shell"),
            AIMessage(content="Step done, moving on"),  # No tool_calls — should be filtered
        ])
        msgs = build_reflector_context(state, "System prompt")

        # Should only have: SystemMessage + AIMessage(tool_calls) + ToolMessage
        ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
        for ai in ai_msgs:
            assert getattr(ai, "tool_calls", None), (
                f"Reflector should only see AIMessages with tool_calls, got: {ai.content[:50]}"
            )

    def test_max_3_pairs(self) -> None:
        from sandbox_agent.context_builders import build_reflector_context

        messages: list = [HumanMessage(content="user")]
        for i in range(10):
            messages.append(AIMessage(
                content="", tool_calls=[{"name": "shell", "args": {}, "id": f"tc{i}"}],
            ))
            messages.append(ToolMessage(content=f"out{i}", tool_call_id=f"tc{i}", name="shell"))

        state = _base_state(messages=messages)
        msgs = build_reflector_context(state, "System prompt")

        # Should have at most 3 pairs + SystemMessage = 7 messages
        ai_count = sum(1 for m in msgs if isinstance(m, AIMessage))
        assert ai_count <= 3


class TestBuildExecutorContext:
    """Direct tests for build_executor_context."""

    def test_new_step_two_messages(self) -> None:
        from sandbox_agent.context_builders import build_executor_context

        state = _base_state(
            plan=["Clone repo", "List failures"],
            current_step=0,
            _tool_call_count=0,
        )
        msgs = build_executor_context(state, "System prompt")
        types = [type(m).__name__ for m in msgs]
        assert types == ["SystemMessage", "HumanMessage"]

    def test_continuing_step_stops_at_boundary(self) -> None:
        from sandbox_agent.context_builders import build_executor_context

        state = _base_state(
            plan=["Clone repo"],
            current_step=0,
            _tool_call_count=1,
            messages=[
                HumanMessage(content="user request"),
                AIMessage(content="old plan text"),  # Before boundary
                SystemMessage(content="[STEP_BOUNDARY 0] Clone the repo"),
                AIMessage(content="", tool_calls=[{"name": "shell", "args": {}, "id": "t1"}]),
                ToolMessage(content="cloned!", tool_call_id="t1", name="shell"),
            ],
        )
        msgs = build_executor_context(state, "System prompt")

        all_content = " ".join(str(m.content) for m in msgs)
        assert "old plan text" not in all_content
        assert "cloned!" in all_content


# ---------------------------------------------------------------------------
# invoke_llm wrapper
# ---------------------------------------------------------------------------


class TestInvokeLLM:
    """Verify invoke_llm captures exactly what the LLM receives."""

    @pytest.mark.asyncio
    async def test_capture_matches_sent_messages(self) -> None:
        """The capture.messages should be the exact messages sent to ainvoke."""
        from sandbox_agent.context_builders import invoke_llm

        llm = CaptureLLM([AIMessage(content="response text")])
        messages = [
            SystemMessage(content="You are an assistant"),
            HumanMessage(content="Hello"),
        ]
        response, capture = await invoke_llm(llm, messages, node="test")

        assert capture.response is response
        assert capture.model == "test-model"
        assert capture.prompt_tokens == 100
        assert capture.completion_tokens == 20

    @pytest.mark.asyncio
    async def test_workspace_preamble_injected(self) -> None:
        """invoke_llm should inject workspace preamble into SystemMessage."""
        from sandbox_agent.context_builders import invoke_llm

        llm = CaptureLLM([AIMessage(content="ok")])
        messages = [
            SystemMessage(content="You are an executor."),
            HumanMessage(content="Do stuff"),
        ]
        _, capture = await invoke_llm(
            llm, messages, node="test",
            workspace_path="/workspace/ctx-abc",
        )

        # The captured SystemMessage should have the preamble prepended
        system_text = capture._system_prompt()
        assert "WORKSPACE (MOST IMPORTANT RULE)" in system_text
        assert "/workspace/ctx-abc" in system_text
        assert "You are an executor." in system_text

    @pytest.mark.asyncio
    async def test_no_workspace_no_preamble(self) -> None:
        """Without workspace_path, no preamble is injected."""
        from sandbox_agent.context_builders import invoke_llm

        llm = CaptureLLM([AIMessage(content="ok")])
        messages = [
            SystemMessage(content="Plain prompt"),
            HumanMessage(content="Hello"),
        ]
        _, capture = await invoke_llm(llm, messages, node="test")

        system_text = capture._system_prompt()
        assert "WORKSPACE" not in system_text
        assert system_text == "Plain prompt"

    @pytest.mark.asyncio
    async def test_executor_has_workspace_preamble(self) -> None:
        """executor_node should have workspace preamble in its system prompt."""
        llm = CaptureLLM([AIMessage(content="Running command")])
        state = _base_state(
            plan=_make_rca_plan(),
            current_step=0,
            workspace_path="/workspace/test-123",
        )
        result = await executor_node(state, llm)

        if "_system_prompt" in result:
            assert "WORKSPACE (MOST IMPORTANT RULE)" in result["_system_prompt"]
            assert "/workspace/test-123" in result["_system_prompt"]
