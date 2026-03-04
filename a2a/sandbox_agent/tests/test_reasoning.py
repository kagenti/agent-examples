"""Tests for the plan-execute-reflect reasoning loop.

Validates:
  - _parse_plan extracts numbered steps from various LLM output formats
  - _parse_decision extracts decisions from LLM output
  - planner_node produces a plan from user messages
  - executor_node signals done when steps exhausted
  - reflector_node skips LLM call for single-step plans
  - reflector_node enforces budget limits
  - reporter_node passes through for single-step plans
  - route_reflector routes correctly based on done flag
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from sandbox_agent.budget import AgentBudget
from sandbox_agent.reasoning import (
    _parse_decision,
    _parse_plan,
    executor_node,
    planner_node,
    reflector_node,
    reporter_node,
    route_reflector,
)


# ---------------------------------------------------------------------------
# _parse_plan
# ---------------------------------------------------------------------------


class TestParsePlan:
    """_parse_plan should extract numbered steps from LLM output."""

    def test_simple_numbered_list(self) -> None:
        text = "1. Run ls\n2. Read the file\n3. Write output"
        steps = _parse_plan(text)
        assert len(steps) == 3
        assert steps[0] == "Run ls"
        assert steps[1] == "Read the file"
        assert steps[2] == "Write output"

    def test_single_step(self) -> None:
        text = "1. Run `ls -la` in the workspace."
        steps = _parse_plan(text)
        assert len(steps) == 1
        assert "ls -la" in steps[0]

    def test_parenthesis_numbering(self) -> None:
        text = "1) List files\n2) Read config"
        steps = _parse_plan(text)
        assert len(steps) == 2

    def test_content_block_list(self) -> None:
        content = [
            {"type": "text", "text": "1. Step one\n2. Step two"},
        ]
        steps = _parse_plan(content)
        assert len(steps) == 2

    def test_fallback_for_unparseable(self) -> None:
        text = "Just do it"
        steps = _parse_plan(text)
        assert len(steps) == 1
        assert "Just do it" in steps[0]

    def test_empty_string_fallback(self) -> None:
        steps = _parse_plan("")
        assert len(steps) == 1

    def test_ignores_non_numbered_lines(self) -> None:
        text = "Here's my plan:\n1. First step\nSome explanation\n2. Second step"
        steps = _parse_plan(text)
        assert len(steps) == 2


# ---------------------------------------------------------------------------
# _parse_decision
# ---------------------------------------------------------------------------


class TestParseDecision:
    """_parse_decision should extract decision words from LLM output."""

    def test_continue(self) -> None:
        assert _parse_decision("continue") == "continue"

    def test_done(self) -> None:
        assert _parse_decision("done") == "done"

    def test_replan(self) -> None:
        assert _parse_decision("replan") == "replan"

    def test_hitl(self) -> None:
        assert _parse_decision("hitl") == "hitl"

    def test_case_insensitive(self) -> None:
        assert _parse_decision("DONE") == "done"
        assert _parse_decision("Continue") == "continue"

    def test_embedded_in_text(self) -> None:
        assert _parse_decision("I think we should continue to the next step") == "continue"

    def test_done_takes_priority(self) -> None:
        # "done" appears before "continue" in the search order
        assert _parse_decision("We are done, no need to continue") == "done"

    def test_default_is_continue(self) -> None:
        assert _parse_decision("some random text") == "continue"

    def test_content_block_list(self) -> None:
        content = [{"type": "text", "text": "done"}]
        assert _parse_decision(content) == "done"


# ---------------------------------------------------------------------------
# route_reflector
# ---------------------------------------------------------------------------


class TestRouteReflector:
    """route_reflector should route based on done flag."""

    def test_done_routes_to_done(self) -> None:
        assert route_reflector({"done": True}) == "done"

    def test_not_done_routes_to_continue(self) -> None:
        assert route_reflector({"done": False}) == "continue"

    def test_missing_done_routes_to_continue(self) -> None:
        assert route_reflector({}) == "continue"


# ---------------------------------------------------------------------------
# planner_node
# ---------------------------------------------------------------------------


class TestPlannerNode:
    """planner_node should produce a plan from user messages."""

    @pytest.mark.asyncio
    async def test_produces_plan(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="1. List files\n2. Read config")

        state = {
            "messages": [HumanMessage(content="set up a project")],
            "iteration": 0,
            "step_results": [],
        }
        result = await planner_node(state, mock_llm)

        assert result["plan"] == ["List files", "Read config"]
        assert result["current_step"] == 0
        assert result["iteration"] == 1
        assert result["done"] is False

    @pytest.mark.asyncio
    async def test_replan_includes_prior_results(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="1. Fix the error")

        state = {
            "messages": [HumanMessage(content="set up a project")],
            "iteration": 1,
            "step_results": ["Step 1 failed: permission denied"],
        }
        result = await planner_node(state, mock_llm)

        # Verify the system message included prior results context
        call_args = mock_llm.ainvoke.call_args[0][0]
        system_text = call_args[0].content
        assert "Previous step results" in system_text
        assert result["iteration"] == 2


# ---------------------------------------------------------------------------
# executor_node
# ---------------------------------------------------------------------------


class TestExecutorNode:
    """executor_node should execute the current plan step."""

    @pytest.mark.asyncio
    async def test_executes_current_step(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="Listed files successfully")

        state = {
            "messages": [HumanMessage(content="set up a project")],
            "plan": ["List files", "Read config"],
            "current_step": 0,
        }
        result = await executor_node(state, mock_llm)

        assert "messages" in result
        # Verify the system prompt mentions step 1
        call_args = mock_llm.ainvoke.call_args[0][0]
        system_text = call_args[0].content
        assert "step 1" in system_text.lower()
        assert "List files" in system_text

    @pytest.mark.asyncio
    async def test_signals_done_when_no_more_steps(self) -> None:
        mock_llm = AsyncMock()

        state = {
            "messages": [HumanMessage(content="test")],
            "plan": ["Only step"],
            "current_step": 1,  # past the only step
        }
        result = await executor_node(state, mock_llm)

        assert result["done"] is True
        mock_llm.ainvoke.assert_not_awaited()


# ---------------------------------------------------------------------------
# reflector_node
# ---------------------------------------------------------------------------


class TestReflectorNode:
    """reflector_node should review output and decide next action."""

    @pytest.mark.asyncio
    async def test_skips_llm_for_single_step(self) -> None:
        mock_llm = AsyncMock()

        state = {
            "messages": [AIMessage(content="Done listing files")],
            "plan": ["List files"],
            "current_step": 0,
            "step_results": [],
            "iteration": 1,
            "done": False,
        }
        result = await reflector_node(state, mock_llm)

        assert result["done"] is True
        mock_llm.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_done_when_executor_signals(self) -> None:
        mock_llm = AsyncMock()

        state = {
            "messages": [],
            "plan": ["Step 1", "Step 2"],
            "current_step": 0,
            "step_results": [],
            "iteration": 1,
            "done": True,
        }
        result = await reflector_node(state, mock_llm)

        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_continues_on_multi_step(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="continue")

        state = {
            "messages": [AIMessage(content="Step 1 completed")],
            "plan": ["Step one", "Step two", "Step three"],
            "current_step": 0,
            "step_results": [],
            "iteration": 1,
            "done": False,
        }
        result = await reflector_node(state, mock_llm)

        assert result["done"] is False
        assert result["current_step"] == 1

    @pytest.mark.asyncio
    async def test_budget_forces_done(self) -> None:
        mock_llm = AsyncMock()
        budget = AgentBudget(max_iterations=2)

        state = {
            "messages": [AIMessage(content="Step result")],
            "plan": ["Step 1", "Step 2", "Step 3"],
            "current_step": 0,
            "step_results": [],
            "iteration": 3,  # exceeds max_iterations=2
            "done": False,
        }
        result = await reflector_node(state, mock_llm, budget=budget)

        assert result["done"] is True
        mock_llm.ainvoke.assert_not_awaited()


# ---------------------------------------------------------------------------
# reporter_node
# ---------------------------------------------------------------------------


class TestReporterNode:
    """reporter_node should format results into a final answer."""

    @pytest.mark.asyncio
    async def test_passthrough_for_single_step(self) -> None:
        mock_llm = AsyncMock()

        state = {
            "messages": [AIMessage(content="file1.txt  file2.txt")],
            "plan": ["List files"],
            "step_results": ["file1.txt  file2.txt"],
        }
        result = await reporter_node(state, mock_llm)

        assert "file1.txt" in result["final_answer"]
        mock_llm.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_summarizes_multi_step(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(
            content="Project setup complete with all tests passing."
        )

        state = {
            "messages": [HumanMessage(content="set up project")],
            "plan": ["Create dirs", "Write code", "Run tests"],
            "step_results": [
                "Created src/ and tests/",
                "Wrote main.py",
                "All tests pass",
            ],
        }
        result = await reporter_node(state, mock_llm)

        assert "Project setup complete" in result["final_answer"]
        mock_llm.ainvoke.assert_awaited_once()
