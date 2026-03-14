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
        AI→Tool message pairs + HumanMessage(reflection prompt).
        The reflection prompt at the END forces the LLM to think about
        the results before calling the next tool.
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
        # Continuing: walk back to [STEP_BOUNDARY N] SystemMessage,
        # then inject a HumanMessage reflection after EACH ToolMessage.
        raw_windowed: list[BaseMessage] = []
        used_chars = 0
        for m in reversed(all_msgs):
            content = str(getattr(m, "content", ""))
            if isinstance(m, SystemMessage) and content.startswith(
                f"[STEP_BOUNDARY {current_step}]"
            ):
                break
            msg_chars = len(content)
            if used_chars + msg_chars > _MAX_CONTEXT_CHARS:
                break
            raw_windowed.insert(0, m)
            used_chars += msg_chars

        # Inject reflection HumanMessage after each ToolMessage
        windowed = []
        call_num = 0
        for m in raw_windowed:
            windowed.append(m)
            if isinstance(m, ToolMessage):
                call_num += 1
                tool_name = getattr(m, "name", "unknown")
                content = str(getattr(m, "content", ""))
                # Determine status from exit code
                if "EXIT_CODE:" in content:
                    import re as _re
                    ec_match = _re.search(r"EXIT_CODE:\s*(\d+)", content)
                    status = "FAILED" if ec_match and ec_match.group(1) != "0" else "OK"
                    error_hint = content[:150] if status == "FAILED" else ""
                elif content.startswith("Error:") or "Permission denied" in content:
                    status = "FAILED"
                    error_hint = content[:150]
                else:
                    status = "OK"
                    error_hint = ""

                reflection_parts = [
                    f"Tool '{tool_name}' call {call_num} {status}.",
                ]
                if error_hint:
                    reflection_parts.append(f"Error: {error_hint}")
                if "unknown flag" in content.lower() or "invalid option" in content.lower():
                    reflection_parts.append(
                        "The flag is INVALID. Run the command with --help to see valid flags."
                    )
                reflection_parts.append(
                    f"Goal: \"{step_text[:100]}\"\n"
                    f"If goal ACHIEVED → stop, summarize result. "
                    f"If FAILED → try DIFFERENT approach. "
                    f"NEVER repeat same command."
                )
                windowed.append(HumanMessage(content=" ".join(reflection_parts)))

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
    bound_tools: list = field(default_factory=list)  # tool schemas sent to LLM

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
        result: dict[str, Any] = {
            "_system_prompt": self._system_prompt()[:10000],
            "_prompt_messages": self._summarize_messages(),
            "_llm_response": self._format_response(),
        }
        if self.bound_tools:
            result["_bound_tools"] = self.bound_tools[:50]
        return result

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
        """Summarize messages as {role, preview} dicts.

        Skips the first SystemMessage since it's already shown as _system_prompt.
        """
        result = []
        skip_first_system = True
        for msg in self.messages:
            if skip_first_system and isinstance(msg, SystemMessage):
                skip_first_system = False
                continue
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


def _extract_bound_tools(llm: Any) -> list[dict[str, Any]]:
    """Extract tool schemas from a LangChain RunnableBinding."""
    try:
        tools = getattr(llm, "kwargs", {}).get("tools", [])
        if not tools:
            first = getattr(llm, "first", None)
            if first:
                tools = getattr(first, "kwargs", {}).get("tools", [])
        result = []
        for t in tools[:50]:
            if isinstance(t, dict):
                fn = t.get("function", t)
                result.append({"name": fn.get("name", "?"), "description": fn.get("description", "")[:100]})
            elif hasattr(t, "name"):
                result.append({"name": t.name, "description": getattr(t, "description", "")[:100]})
        return result
    except Exception:
        return []


async def invoke_llm(
    llm: Any,
    messages: list[BaseMessage],
    *,
    node: str = "",
    session_id: str = "",
    workspace_path: str = "",
) -> tuple[AIMessage, LLMCallCapture]:
    """Invoke the LLM and capture the exact input/output.

    If ``workspace_path`` is provided, the workspace preamble is
    automatically prepended to the first SystemMessage. This ensures
    every LLM call sees the workspace path rule — nodes don't need
    to inject it manually.

    Returns ``(response, capture)`` where capture contains:
    - ``messages``: the exact messages sent to the LLM (with preamble)
    - ``response``: the AIMessage returned
    - ``prompt_tokens`` / ``completion_tokens``: token usage
    - ``model``: model name from response metadata

    Usage in a node::

        messages = build_executor_context(state, system_content)
        response, capture = await invoke_llm(
            llm, messages, node="executor",
            workspace_path=state.get("workspace_path", "/workspace"),
        )
    """
    # Inject workspace preamble into the first SystemMessage
    if workspace_path and messages:
        from sandbox_agent.prompts import WORKSPACE_PREAMBLE

        preamble = WORKSPACE_PREAMBLE.format(workspace_path=workspace_path)
        if isinstance(messages[0], SystemMessage):
            messages = [
                SystemMessage(content=preamble + "\n" + messages[0].content),
                *messages[1:],
            ]
        else:
            # No SystemMessage — prepend one
            messages = [SystemMessage(content=preamble), *messages]

    response = await llm.ainvoke(messages)

    usage = getattr(response, "usage_metadata", None) or {}
    prompt_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
    model_name = (getattr(response, "response_metadata", None) or {}).get("model", "")

    # Extract bound tools from the LLM (RunnableBinding stores them in kwargs)
    bound_tools = _extract_bound_tools(llm)

    capture = LLMCallCapture(
        messages=list(messages),
        response=response,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model_name,
        bound_tools=bound_tools,
    )

    logger.info(
        "LLM call [%s]: %d messages, %d prompt tokens, %d completion tokens, model=%s",
        node, len(messages), prompt_tokens, completion_tokens, model_name,
        extra={"session_id": session_id, "node": node,
               "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    )

    return response, capture


def _build_tool_descriptions(llm_with_tools: Any) -> str:
    """Build a textual description of bound tools for the thinking prompt."""
    tools = _extract_bound_tools(llm_with_tools)
    if not tools:
        return ""
    lines = ["Available tools:"]
    for t in tools:
        name = t.get("name", "?")
        desc = t.get("description", "")
        lines.append(f"  - {name}: {desc}" if desc else f"  - {name}")
    return "\n".join(lines)


async def invoke_with_tool_loop(
    llm_with_tools: Any,
    llm_reason: Any | None,
    messages: list[BaseMessage],
    *,
    node: str,
    session_id: str,
    workspace_path: str,
    thinking_budget: int = 5,
    max_parallel_tool_calls: int = 5,
) -> tuple[AIMessage, LLMCallCapture, list[dict[str, Any]]]:
    """Invoke LLM with optional thinking iterations + micro-reasoning.

    Returns ``(response, capture, sub_events)`` where sub_events is a list
    of thinking event dicts — one per thinking iteration.

    When ``llm_reason`` is provided (thinking mode):
      1. Thinking loop (up to ``thinking_budget`` iterations):
         Bare LLM reasons about what to do. Each iteration sees previous
         thinking texts and tool descriptions (no actual tool bindings).
      2. Micro-reasoning: LLM with tools (tool_choice=any) makes tool calls.
         Allows up to ``max_parallel_tool_calls`` parallel calls.

    Each thinking sub_event has full debug data (system_prompt, prompt_messages,
    bound_tools, llm_response) so the UI can inspect every call.

    When ``llm_reason`` is None (single-phase mode):
      One call to llm_with_tools with implicit auto. No sub_events.
    """
    sub_events: list[dict[str, Any]] = []

    if llm_reason is not None:
        # Build textual tool descriptions for the thinking prompt
        tool_desc_text = _build_tool_descriptions(llm_with_tools)

        # Thinking loop: up to thinking_budget bare LLM iterations
        thinking_history: list[BaseMessage] = []
        total_thinking_tokens = 0
        last_reasoning = ""

        for i in range(thinking_budget):
            # Build thinking messages: original messages + tool descriptions + thinking history
            thinking_messages = list(messages)

            # Inject tool descriptions into the system message
            if tool_desc_text and thinking_messages and isinstance(thinking_messages[0], SystemMessage):
                thinking_messages[0] = SystemMessage(
                    content=thinking_messages[0].content + "\n\n" + tool_desc_text
                )

            # Add thinking history from previous iterations
            thinking_messages.extend(thinking_history)

            # Add thinking prompt
            if i == 0:
                thinking_messages.append(
                    HumanMessage(content="Think step by step about what to do. "
                                 "Reason about the best approach before acting. "
                                 "Do NOT call any tools — just think.")
                )
            else:
                thinking_messages.append(
                    HumanMessage(content="Continue thinking. Refine your approach "
                                 "based on your previous reasoning. "
                                 "When ready to act, start with 'READY:' followed by your action plan.")
                )

            reason_response, reason_capture = await invoke_llm(
                llm_reason, thinking_messages,
                node=f"{node}-think-{i+1}", session_id=session_id,
                workspace_path=workspace_path,
            )
            last_reasoning = str(reason_response.content or "").strip()
            total_thinking_tokens += reason_capture.prompt_tokens + reason_capture.completion_tokens

            # Emit thinking iteration as a sub_event with full debug data
            sub_events.append({
                "type": "thinking",
                "node": node,
                "iteration": i + 1,
                "total_iterations": 0,  # updated after loop
                "reasoning": last_reasoning,
                **reason_capture.debug_fields(),
                **reason_capture.token_fields(),
            })

            # Add to thinking history for next iteration
            thinking_history.extend([
                AIMessage(content=last_reasoning),
                HumanMessage(content=f"(Thinking {i+1} recorded. Continue or signal READY:)"),
            ])

            # Early break if LLM signals readiness
            if last_reasoning.upper().startswith("READY:"):
                break

        # Update total_iterations on all sub_events
        total_iters = len(sub_events)
        for evt in sub_events:
            evt["total_iterations"] = total_iters

        logger.info(
            "Thinking %s: %d iterations, %d tokens",
            node, total_iters, total_thinking_tokens,
            extra={"session_id": session_id, "node": node,
                   "thinking_iterations": total_iters},
        )

        # Micro-reasoning: LLM with tools makes the actual tool call(s)
        # Include last thinking text as context
        tool_messages = messages + [
            AIMessage(content=last_reasoning or "I need to call a tool for this step."),
            HumanMessage(content="Now execute the action you described above. "
                         f"Call up to {max_parallel_tool_calls} tools."),
        ]
        response, capture = await invoke_llm(
            llm_with_tools, tool_messages,
            node=f"{node}-tool", session_id=session_id,
            workspace_path=workspace_path,
        )
        # Merge all thinking tokens into the capture
        capture.prompt_tokens += total_thinking_tokens
        capture.completion_tokens += 0  # thinking completion tokens already counted

        # If micro-reasoning produced tool calls but no text, merge last thinking
        if last_reasoning and response.tool_calls and not response.content:
            response = AIMessage(
                content=last_reasoning,
                tool_calls=response.tool_calls,
            )

        # Enforce max parallel tool calls
        if len(response.tool_calls) > max_parallel_tool_calls:
            logger.info(
                "Micro-reasoning returned %d tool calls — keeping first %d",
                len(response.tool_calls), max_parallel_tool_calls,
                extra={"session_id": session_id, "node": node},
            )
            response = AIMessage(
                content=response.content,
                tool_calls=response.tool_calls[:max_parallel_tool_calls],
            )

        logger.info(
            "Think-act %s: %d thinking + micro-reasoning → %d tool calls",
            node, total_iters, len(response.tool_calls),
            extra={"session_id": session_id, "node": node},
        )
    else:
        # Single-phase: one LLM call with implicit auto
        response, capture = await invoke_llm(
            llm_with_tools, messages,
            node=node, session_id=session_id,
            workspace_path=workspace_path,
        )

    return response, capture, sub_events
