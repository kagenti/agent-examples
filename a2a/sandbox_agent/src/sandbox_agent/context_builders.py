"""Pure functions that build the message list for each reasoning node,
and an ``invoke_llm`` wrapper that guarantees the debug output matches
exactly what was sent to the LLM.

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

import json
import logging
import os
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# LLM invocation wrapper — captures exactly what the LLM sees
# ---------------------------------------------------------------------------

_DEBUG_PROMPTS = os.environ.get("SANDBOX_DEBUG_PROMPTS", "1") == "1"


@dataclass
class LLMCallCapture:
    """Captures the exact input/output of an LLM invocation.

    Always populated (not conditional on _DEBUG_PROMPTS) so that the
    node result can decide what to include.  This guarantees the debug
    view shows exactly what the LLM received — no drift.
    """

    messages: list = field(default_factory=list)
    response: Any = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    # -- Convenience methods for node result dicts -------------------------

    def debug_fields(self) -> dict[str, Any]:
        """Return prompt debug fields for the node result dict.

        Only populated when ``SANDBOX_DEBUG_PROMPTS=1`` (default).
        These are large payloads (system prompt, message list, full
        response) — optional to reduce event size in production.
        Token counts and budget are always included via ``token_fields()``.
        """
        if not _DEBUG_PROMPTS:
            return {}
        return {
            "_system_prompt": self._system_prompt()[:10000],
            "_prompt_messages": self._summarize_messages(),
            "_llm_response": self._format_response(),
        }

    def token_fields(self) -> dict[str, Any]:
        """Return token usage fields for the node result dict."""
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    # -- Internal helpers --------------------------------------------------

    def _system_prompt(self) -> str:
        """Extract the system prompt from the captured messages."""
        for m in self.messages:
            if isinstance(m, SystemMessage):
                return str(m.content)
        return ""

    def _summarize_messages(self) -> list[dict[str, str]]:
        """Summarize messages as {role, preview} dicts."""
        result = []
        for msg in self.messages:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            text = str(content)
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                tc_parts = []
                for tc in tool_calls:
                    name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
                    args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    args_str = str(args)[:500] if args else ""
                    tc_parts.append(f"{name}({args_str})" if args_str else name)
                text = f"[tool_calls: {'; '.join(tc_parts)}] {text[:2000]}"
            tool_name = getattr(msg, "name", None)
            if role == "tool" and tool_name:
                text = f"[{tool_name}] {text[:3000]}"
            else:
                text = text[:5000]
            result.append({"role": role, "preview": text})
        return result

    def _format_response(self) -> dict[str, Any]:
        """Format the LLM response as OpenAI-style dict."""
        resp = self.response
        if resp is None:
            return {}
        try:
            meta = getattr(resp, "response_metadata", {}) or {}
            content = resp.content
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ) or None
            tool_calls_out = None
            if resp.tool_calls:
                tool_calls_out = [
                    {
                        "id": tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?"),
                            "arguments": json.dumps(
                                tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                            ),
                        },
                    }
                    for tc in resp.tool_calls
                ]
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content if content else None,
                        "tool_calls": tool_calls_out,
                    },
                    "finish_reason": meta.get("finish_reason", "unknown"),
                }],
                "model": meta.get("model", ""),
                "usage": {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                },
                "id": meta.get("id", ""),
            }
        except Exception:
            return {"error": "Failed to format response"}


async def invoke_llm(
    llm: Any,
    messages: list[BaseMessage],
    *,
    node: str = "",
    session_id: str = "",
) -> tuple[AIMessage, LLMCallCapture]:
    """Invoke the LLM and capture the exact input/output.

    Returns ``(response, capture)`` where capture contains:
    - ``messages``: the exact messages sent to the LLM
    - ``response``: the AIMessage returned
    - ``prompt_tokens`` / ``completion_tokens``: token usage
    - ``model``: model name from response metadata

    Usage in a node::

        messages = build_executor_context(state, system_content)
        response, capture = await invoke_llm(llm, messages, node="executor")
        result = {
            "messages": [response],
            **capture.token_fields(),
            **capture.debug_fields(),
        }
    """
    response = await llm.ainvoke(messages)

    usage = getattr(response, "usage_metadata", None) or {}
    prompt_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
    model_name = (getattr(response, "response_metadata", None) or {}).get("model", "")

    capture = LLMCallCapture(
        messages=list(messages),
        response=response,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model_name,
    )

    logger.info(
        "LLM call [%s]: %d messages, %d prompt tokens, %d completion tokens, model=%s",
        node, len(messages), prompt_tokens, completion_tokens, model_name,
        extra={"session_id": session_id, "node": node,
               "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    )

    return response, capture
