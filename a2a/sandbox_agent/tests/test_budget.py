"""Tests for AgentBudget tracking.

Validates:
  - Default limits are sensible
  - Counters increment correctly
  - Exceeded properties trigger at the right thresholds
  - HITL check-in fires on the correct interval
  - Per-step tool counter resets between steps
"""

from __future__ import annotations

from sandbox_agent.budget import AgentBudget


class TestDefaults:
    """Default budget values should match the design spec."""

    def test_default_max_iterations(self) -> None:
        b = AgentBudget()
        assert b.max_iterations == 10

    def test_default_max_tool_calls_per_step(self) -> None:
        b = AgentBudget()
        assert b.max_tool_calls_per_step == 5

    def test_default_max_tokens(self) -> None:
        b = AgentBudget()
        assert b.max_tokens == 200_000

    def test_default_hitl_interval(self) -> None:
        b = AgentBudget()
        assert b.hitl_interval == 5

    def test_counters_start_at_zero(self) -> None:
        b = AgentBudget()
        assert b.iterations_used == 0
        assert b.tokens_used == 0
        assert b.tool_calls_this_step == 0


class TestIterations:
    """Iteration tracking and exceeded detection."""

    def test_tick_increments(self) -> None:
        b = AgentBudget(max_iterations=3)
        b.tick_iteration()
        assert b.iterations_used == 1
        b.tick_iteration()
        assert b.iterations_used == 2

    def test_not_exceeded_before_limit(self) -> None:
        b = AgentBudget(max_iterations=3)
        b.tick_iteration()
        b.tick_iteration()
        assert not b.iterations_exceeded

    def test_exceeded_at_limit(self) -> None:
        b = AgentBudget(max_iterations=3)
        for _ in range(3):
            b.tick_iteration()
        assert b.iterations_exceeded

    def test_exceeded_propagates_to_overall(self) -> None:
        b = AgentBudget(max_iterations=1)
        assert not b.exceeded
        b.tick_iteration()
        assert b.exceeded


class TestTokens:
    """Token tracking and exceeded detection."""

    def test_add_tokens(self) -> None:
        b = AgentBudget(max_tokens=1000)
        b.add_tokens(500)
        assert b.tokens_used == 500
        b.add_tokens(300)
        assert b.tokens_used == 800

    def test_not_exceeded_below_limit(self) -> None:
        b = AgentBudget(max_tokens=1000)
        b.add_tokens(999)
        assert not b.tokens_exceeded

    def test_exceeded_at_limit(self) -> None:
        b = AgentBudget(max_tokens=1000)
        b.add_tokens(1000)
        assert b.tokens_exceeded

    def test_exceeded_propagates_to_overall(self) -> None:
        b = AgentBudget(max_tokens=100)
        b.add_tokens(200)
        assert b.exceeded


class TestStepTools:
    """Per-step tool-call tracking."""

    def test_tick_tool_call(self) -> None:
        b = AgentBudget(max_tool_calls_per_step=3)
        b.tick_tool_call()
        assert b.tool_calls_this_step == 1

    def test_not_exceeded_below(self) -> None:
        b = AgentBudget(max_tool_calls_per_step=3)
        b.tick_tool_call()
        b.tick_tool_call()
        assert not b.step_tools_exceeded

    def test_exceeded_at_limit(self) -> None:
        b = AgentBudget(max_tool_calls_per_step=2)
        b.tick_tool_call()
        b.tick_tool_call()
        assert b.step_tools_exceeded

    def test_reset_clears_counter(self) -> None:
        b = AgentBudget(max_tool_calls_per_step=2)
        b.tick_tool_call()
        b.tick_tool_call()
        assert b.step_tools_exceeded
        b.reset_step_tools()
        assert b.tool_calls_this_step == 0
        assert not b.step_tools_exceeded


class TestHitlCheckin:
    """HITL check-in fires at the configured interval."""

    def test_no_checkin_at_zero(self) -> None:
        b = AgentBudget(hitl_interval=5)
        assert not b.needs_hitl_checkin

    def test_checkin_at_interval(self) -> None:
        b = AgentBudget(hitl_interval=3)
        for _ in range(3):
            b.tick_iteration()
        assert b.needs_hitl_checkin

    def test_no_checkin_between_intervals(self) -> None:
        b = AgentBudget(hitl_interval=3)
        b.tick_iteration()
        assert not b.needs_hitl_checkin
        b.tick_iteration()
        assert not b.needs_hitl_checkin

    def test_disabled_when_zero_interval(self) -> None:
        b = AgentBudget(hitl_interval=0)
        b.tick_iteration()
        assert not b.needs_hitl_checkin
