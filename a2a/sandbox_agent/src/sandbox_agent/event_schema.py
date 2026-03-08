# Copyright 2025 IBM Corp.
# Licensed under the Apache License, Version 2.0

"""Typed event schema for LangGraph node events.

Each LangGraph node emits a distinct event type. The dataclasses here are
the single source of truth; the TypeScript frontend mirrors these types
in ``agentLoop.ts``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List


class NodeEventType:
    """Constants for the ``type`` discriminator on every LoopEvent."""

    PLANNER_OUTPUT = "planner_output"
    EXECUTOR_STEP = "executor_step"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REFLECTOR_DECISION = "reflector_decision"
    REPORTER_OUTPUT = "reporter_output"
    BUDGET_UPDATE = "budget_update"
    HITL_REQUEST = "hitl_request"


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


@dataclass
class LoopEvent:
    """Base event emitted by a graph node during the reasoning loop."""

    type: str  # One of NodeEventType constants
    loop_id: str  # Unique per reasoning loop invocation
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))


# ---------------------------------------------------------------------------
# Concrete event types
# ---------------------------------------------------------------------------


@dataclass
class PlannerOutput(LoopEvent):
    """Planner created or revised a plan."""

    type: str = NodeEventType.PLANNER_OUTPUT
    steps: List[str] = field(default_factory=list)
    iteration: int = 0


@dataclass
class ExecutorStep(LoopEvent):
    """Executor is working on a plan step."""

    type: str = NodeEventType.EXECUTOR_STEP
    step: int = 0
    total_steps: int = 0
    description: str = ""


@dataclass
class ToolCall(LoopEvent):
    """Executor invoked a tool."""

    type: str = NodeEventType.TOOL_CALL
    step: int = 0
    name: str = ""
    args: str = ""


@dataclass
class ToolResult(LoopEvent):
    """Tool returned a result."""

    type: str = NodeEventType.TOOL_RESULT
    step: int = 0
    name: str = ""
    output: str = ""


@dataclass
class ReflectorDecision(LoopEvent):
    """Reflector reviewed execution and decided next action."""

    type: str = NodeEventType.REFLECTOR_DECISION
    decision: str = ""  # "continue", "replan", "done"
    assessment: str = ""  # Full reflection text
    iteration: int = 0


@dataclass
class ReporterOutput(LoopEvent):
    """Reporter generated the final answer."""

    type: str = NodeEventType.REPORTER_OUTPUT
    content: str = ""


@dataclass
class BudgetUpdate(LoopEvent):
    """Budget tracking update."""

    type: str = NodeEventType.BUDGET_UPDATE
    tokens_used: int = 0
    tokens_budget: int = 0
    wall_clock_s: float = 0
    max_wall_clock_s: float = 0
