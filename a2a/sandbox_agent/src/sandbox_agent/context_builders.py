"""Pure functions that build the message list for each reasoning node.

Each builder takes the graph state and returns a list of BaseMessage objects
that the node should pass to ``llm.ainvoke()``.  The functions are
independently testable and enforce context isolation — no node sees
messages it shouldn't.

Context contracts:

    Planner   — SystemMessage(prompt + step status) + HumanMessage(user request only).
                Does NOT include own previous AIMessages (prevents replan duplication).
    Executor  — SystemMessage(prompt) + HumanMessage(step brief) + this step's tool pairs.
                Stops at [STEP_BOUNDARY] SystemMessage. Never sees planner output.
    Reflector — SystemMessage(prompt) + last 3 tool-call AI→Tool pairs.
                Filters out non-tool AIMessages (planner/reflector text).
    Reporter  — SystemMessage(prompt) + full history (intentional for summarization).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Planner context
# ---------------------------------------------------------------------------

_MAX_PLANNER_HISTORY_MSGS = 6  # user request + a few recent tool results


def build_planner_context(
    state: dict[str, Any],
    system_content: str,
) -> list[BaseMessage]:
    """Build the message list for the planner node.

    On fresh plan (iteration 0): SystemMessage + all user HumanMessages.
    On replan (iteration > 0): SystemMessage + user request + last few
    ToolMessages for context.  **Excludes** previous planner AIMessages
    to prevent the LLM from seeing and duplicating its own plan.

    The step status and tool history are already in ``system_content``
    (built by the caller), so they don't need to appear as messages.
    """
    messages = state.get("messages", [])
    iteration = state.get("iteration", 0)

    if iteration == 0:
        # Fresh plan: include only HumanMessages (user requests)
        user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        return [SystemMessage(content=system_content)] + user_msgs

    # Replan: user request + last few tool results for context.
    # Explicitly EXCLUDE previous planner AIMessages to prevent duplication.
    user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    # Take the first user message (original request)
    first_user = user_msgs[:1] if user_msgs else []

    # Include last few ToolMessages so planner knows what was tried
    recent_tools: list[BaseMessage] = []
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            recent_tools.insert(0, m)
            if len(recent_tools) >= _MAX_PLANNER_HISTORY_MSGS:
                break

    result = [SystemMessage(content=system_content)] + first_user + recent_tools
    logger.info(
        "Planner context: %d messages (iteration=%d, %d tool results)",
        len(result), iteration, len(recent_tools),
        extra={"session_id": state.get("context_id", ""), "node": "planner"},
    )
    return result


# ---------------------------------------------------------------------------
# Executor context
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4
_MAX_CONTEXT_CHARS = 30_000 * _CHARS_PER_TOKEN  # ~120k chars


def build_executor_context(
    state: dict[str, Any],
    system_content: str,
) -> list[BaseMessage]:
    """Build the message list for the executor node.

    On new step (tool_call_count == 0):
        SystemMessage(prompt) + HumanMessage(step brief).
        The executor sees ONLY the step description — no plan, no history.

    On continuing step (tool_call_count > 0):
        SystemMessage(prompt) + HumanMessage(step brief) + this step's
        AI→Tool message pairs.  Walks backward from the end of messages,
        stopping at the [STEP_BOUNDARY] SystemMessage.  Capped at ~30k
        tokens to stay within context window.
    """
    all_msgs = state.get("messages", [])
    current_step = state.get("current_step", 0)
    tool_call_count = state.get("_tool_call_count", 0)
    plan = state.get("plan", [])
    step_text = plan[current_step] if current_step < len(plan) else "N/A"
    step_brief = state.get(
        "skill_instructions",
        f"Execute step {current_step + 1}: {step_text}",
    )

    first_msg = [HumanMessage(content=step_brief)]

    if tool_call_count == 0:
        # New step: only the step brief
        windowed: list[BaseMessage] = []
    else:
        # Continuing: walk back to [STEP_BOUNDARY N] SystemMessage
        windowed = []
        used_chars = 0
        for m in reversed(all_msgs):
            content = str(getattr(m, "content", ""))
            # Stop at the SystemMessage boundary for this step
            if isinstance(m, SystemMessage) and content.startswith(
                f"[STEP_BOUNDARY {current_step}]"
            ):
                break
            msg_chars = len(content)
            if used_chars + msg_chars > _MAX_CONTEXT_CHARS:
                break
            windowed.insert(0, m)
            used_chars += msg_chars

    result = [SystemMessage(content=system_content)] + first_msg + windowed
    logger.info(
        "Executor context: %d messages, ~%dk chars (from %d total)",
        len(result), sum(len(str(getattr(m, "content", ""))) for m in result) // 1000,
        len(all_msgs),
        extra={
            "session_id": state.get("context_id", ""),
            "node": "executor",
            "current_step": current_step,
            "tool_call_count": tool_call_count,
        },
    )
    return result


# ---------------------------------------------------------------------------
# Reflector context
# ---------------------------------------------------------------------------

_MAX_REFLECTOR_PAIRS = 3  # last 3 AI→Tool pairs (6 messages max)


def build_reflector_context(
    state: dict[str, Any],
    system_content: str,
) -> list[BaseMessage]:
    """Build the message list for the reflector node.

    Includes only the last ``_MAX_REFLECTOR_PAIRS`` AI→Tool pairs from
    the message history.  **Filters out** AIMessages that have no
    ``tool_calls`` (planner plan text, reflector decisions, executor
    summaries) to prevent plan leakage.

    The plan text and step results are already in ``system_content``
    (formatted from state fields), so they don't need to appear as
    conversation messages.
    """
    messages = state.get("messages", [])

    recent_msgs: list[BaseMessage] = []
    pair_count = 0
    for m in reversed(messages):
        if isinstance(m, SystemMessage):
            continue
        # Skip AIMessages without tool_calls (planner/reflector text output).
        # These would leak plan context into the reflector.
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            continue
        recent_msgs.insert(0, m)
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            pair_count += 1
            if pair_count >= _MAX_REFLECTOR_PAIRS:
                break

    result = [SystemMessage(content=system_content)] + recent_msgs
    logger.info(
        "Reflector context: %d messages (%d tool pairs from %d total)",
        len(result), pair_count, len(messages),
        extra={"session_id": state.get("context_id", ""), "node": "reflector"},
    )
    return result
