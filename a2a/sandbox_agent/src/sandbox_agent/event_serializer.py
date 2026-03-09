"""Framework-specific event serializers for structured JSON streaming.

Each agent framework (LangGraph, CrewAI, AG2) has its own internal event
format. Serializers convert framework events into a common JSON schema
that the backend and frontend understand.

Event types (new — node-specific):
    planner_output     — Planner created/revised a plan
    executor_step      — Executor starts working on a plan step
    tool_call          — Tool invoked (unchanged)
    tool_result        — Tool returned output (unchanged)
    reflector_decision — Reflector decides continue/replan/done
    reporter_output    — Reporter generates the final answer
    budget_update      — Budget tracking
    error              — An error occurred during execution
    hitl_request       — Human-in-the-loop approval is needed

Legacy types (kept for backward compatibility):
    plan          — Alias for planner_output
    plan_step     — Alias for executor_step
    reflection    — Alias for reflector_decision
    llm_response  — Generic LLM text (used for unknown nodes only)
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


def _safe_tc(tc: Any) -> dict[str, Any]:
    """Safely extract name/args from a tool call object.

    LangChain tool_calls can be dicts, ToolCall TypedDicts, or
    InvalidToolCall objects (tuples). Handle all formats gracefully.
    """
    try:
        if isinstance(tc, dict):
            return {"name": tc.get("name", "unknown"), "args": tc.get("args", {})}
        if hasattr(tc, "name"):
            return {"name": getattr(tc, "name", "unknown"), "args": getattr(tc, "args", {})}
        if isinstance(tc, (list, tuple)) and len(tc) >= 2:
            return {"name": str(tc[0]), "args": tc[1] if isinstance(tc[1], dict) else {}}
    except Exception:
        pass
    return {"name": "unknown", "args": {}}


class FrameworkEventSerializer(ABC):
    """Base class for framework-specific event serialization.

    Subclass this for each agent framework (LangGraph, CrewAI, AG2).
    The ``serialize`` method must return a JSON string with at least
    a ``type`` field.
    """

    @abstractmethod
    def serialize(self, key: str, value: dict) -> str:
        """Serialize a framework event into a JSON string.

        Parameters
        ----------
        key:
            The graph node name (e.g. "assistant", "tools").
        value:
            The event payload from the framework's streaming API.

        Returns
        -------
        str
            A JSON string with at least ``{"type": "..."}``
        """
        ...


class LangGraphSerializer(FrameworkEventSerializer):
    """Serialize LangGraph ``stream_mode='updates'`` events.

    LangGraph emits events like::

        {"assistant": {"messages": [AIMessage(...)]}}
        {"tools": {"messages": [ToolMessage(...)]}}

    This serializer extracts tool calls, tool results, and LLM
    responses into structured JSON.

    When the graph uses a plan-execute-reflect reasoning loop, all
    events include a ``loop_id`` so the frontend can group them into
    an expandable AgentLoopCard.
    """

    def __init__(self, loop_id: str | None = None, context_id: str | None = None) -> None:
        import uuid
        self._loop_id = loop_id or str(uuid.uuid4())[:8]
        self._step_index = 0
        self._context_id = context_id or "unknown"

    def serialize(self, key: str, value: dict) -> str:
        # Reasoning-loop nodes may emit state fields instead of messages
        if key == "planner":
            result = self._serialize_planner(value)
        elif key == "reflector":
            result = self._serialize_reflector(value)
        elif key == "reporter":
            result = self._serialize_reporter(value)
        else:
            msgs = value.get("messages", [])
            if not msgs:
                result = json.dumps({"type": "llm_response", "content": f"[{key}]"})
            else:
                msg = msgs[-1]

                if key == "executor":
                    result = self._serialize_executor(msg, value)
                elif key == "tools":
                    result = self._serialize_tool_result(msg)
                else:
                    # Unknown node — treat as informational
                    content = getattr(msg, "content", "")
                    if isinstance(content, list):
                        text = self._extract_text_blocks(content)
                    else:
                        text = str(content)[:2000] if content else f"[{key}]"
                    result = json.dumps({"type": "llm_response", "content": text})

        # Log each serialized event for pipeline observability (Stage 1)
        for line in result.split("\n"):
            line = line.strip()
            if line:
                try:
                    event_type = json.loads(line).get("type", "?")
                except json.JSONDecodeError:
                    event_type = "parse_error"
                logger.info("SERIALIZE session=%s loop=%s type=%s step=%s",
                    self._context_id, self._loop_id, event_type, self._step_index)

        return result

    def _serialize_assistant(self, msg: Any) -> str:
        """Serialize an assistant (LLM) node output.

        When the LLM calls tools, it often also produces reasoning text.
        We emit BOTH the thinking content and the tool call as separate
        JSON lines so the UI shows the full chain:
            {"type": "llm_response", "content": "Let me check..."}
            {"type": "tool_call", "tools": [...]}
        """
        tool_calls = getattr(msg, "tool_calls", [])
        content = getattr(msg, "content", "")

        # Extract any text content from the LLM
        if isinstance(content, list):
            text = self._extract_text_blocks(content)
        else:
            text = str(content)[:2000] if content else ""

        if tool_calls:
            parts = []
            # Emit thinking/reasoning text first (if present)
            if text.strip():
                parts.append(json.dumps({"type": "llm_response", "content": text}))
            # Then emit the tool call
            parts.append(json.dumps({
                "type": "tool_call",
                "tools": [
                    _safe_tc(tc)
                    for tc in tool_calls
                ],
            }))
            return "\n".join(parts)

        return json.dumps({"type": "llm_response", "content": text})

    def _serialize_executor(self, msg: Any, value: dict | None = None) -> str:
        """Serialize an executor node output with loop_id for AgentLoopCard."""
        tool_calls = getattr(msg, "tool_calls", [])
        content = getattr(msg, "content", "")

        if isinstance(content, list):
            text = self._extract_text_blocks(content)
        else:
            text = str(content)[:2000] if content else ""

        parts = []

        _v = value or {}
        plan = _v.get("plan", [])
        model = _v.get("model", "")
        prompt_tokens = _v.get("prompt_tokens", 0)
        completion_tokens = _v.get("completion_tokens", 0)

        # Emit executor_step event so UI shows which step is executing
        step_payload = {
            "type": "executor_step",
            "loop_id": self._loop_id,
            "step": self._step_index,
            "total_steps": len(plan) if plan else 0,
            "description": text[:200] if text else "",
            "reasoning": text[:2000] if text else "",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        parts.append(json.dumps(step_payload))
        # Legacy alias for backward compatibility
        parts.append(json.dumps(dict(step_payload, type="plan_step")))

        if tool_calls:
            parts.append(json.dumps({
                "type": "tool_call",
                "loop_id": self._loop_id,
                "step": self._step_index,
                "tools": [
                    _safe_tc(tc)
                    for tc in tool_calls
                ],
            }))
            return "\n".join(parts)

        # Emit tool_call event for text-parsed tools (no structured tool_calls)
        parsed_tools = _v.get("parsed_tools", [])
        if parsed_tools:
            parts.append(json.dumps({
                "type": "tool_call",
                "loop_id": self._loop_id,
                "step": self._step_index,
                "tools": [
                    {"name": t["name"], "args": t.get("args", {})}
                    for t in parsed_tools
                ],
            }))

        return "\n".join(parts)

    def _serialize_tool_result(self, msg: Any) -> str:
        """Serialize a tool node output with loop_id."""
        name = getattr(msg, "name", "unknown")
        content = getattr(msg, "content", "")
        return json.dumps({
            "type": "tool_result",
            "loop_id": self._loop_id,
            "step": self._step_index,
            "name": str(name),
            "output": str(content)[:2000],
        })

    def _serialize_planner(self, value: dict) -> str:
        """Serialize a planner node output — emits planner_output + legacy plan."""
        plan = value.get("plan", [])
        iteration = value.get("iteration", 1)

        # Also include any LLM text from the planner's message
        msgs = value.get("messages", [])
        text = ""
        if msgs:
            content = getattr(msgs[-1], "content", "")
            if isinstance(content, list):
                text = self._extract_text_blocks(content)
            else:
                text = str(content)[:2000] if content else ""

        model = value.get("model", "")
        prompt_tokens = value.get("prompt_tokens", 0)
        completion_tokens = value.get("completion_tokens", 0)

        payload = {
            "type": "planner_output",
            "loop_id": self._loop_id,
            "steps": plan,
            "iteration": iteration,
            "content": text,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

        # Emit new type + legacy type for backward compatibility
        legacy = dict(payload, type="plan")
        return "\n".join([json.dumps(payload), json.dumps(legacy)])

    def _serialize_reflector(self, value: dict) -> str:
        """Serialize a reflector node output — emits reflector_decision + legacy reflection."""
        done = value.get("done", False)
        current_step = value.get("current_step", 0)
        step_results = value.get("step_results", [])

        # Extract decision text from message if present
        msgs = value.get("messages", [])
        text = ""
        if msgs:
            content = getattr(msgs[-1], "content", "")
            if isinstance(content, list):
                text = self._extract_text_blocks(content)
            else:
                text = str(content)[:500] if content else ""

        # Derive the decision keyword from the text
        decision = "done" if done else self._extract_decision(text)

        # Advance step index when reflector completes a step
        self._step_index = current_step

        model = value.get("model", "")
        prompt_tokens = value.get("prompt_tokens", 0)
        completion_tokens = value.get("completion_tokens", 0)
        iteration = value.get("iteration", 0)

        payload = {
            "type": "reflector_decision",
            "loop_id": self._loop_id,
            "decision": decision,
            "assessment": text,
            "iteration": iteration,
            "done": done,
            "current_step": current_step,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

        # Emit new type + legacy type for backward compatibility
        legacy = {
            "type": "reflection",
            "loop_id": self._loop_id,
            "done": done,
            "current_step": current_step,
            "assessment": text,
            "content": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        return "\n".join([json.dumps(payload), json.dumps(legacy)])

    def _serialize_reporter(self, value: dict) -> str:
        """Serialize a reporter node output — emits reporter_output."""
        final_answer = value.get("final_answer", "")

        # Also check messages for the reporter's LLM response
        if not final_answer:
            msgs = value.get("messages", [])
            if msgs:
                content = getattr(msgs[-1], "content", "")
                if isinstance(content, list):
                    final_answer = self._extract_text_blocks(content)
                else:
                    final_answer = str(content)[:2000] if content else ""

        model = value.get("model", "")
        prompt_tokens = value.get("prompt_tokens", 0)
        completion_tokens = value.get("completion_tokens", 0)

        return json.dumps({
            "type": "reporter_output",
            "loop_id": self._loop_id,
            "content": final_answer[:2000],
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })

    @staticmethod
    def _extract_decision(text: str) -> str:
        """Extract a decision keyword from reflector text.

        Returns one of: ``continue``, ``replan``, ``done``, ``hitl``.
        Defaults to ``continue`` if the text is ambiguous.
        """
        text_lower = text.strip().lower()
        for decision in ("done", "replan", "hitl", "continue"):
            if decision in text_lower:
                return decision
        return "continue"

    @staticmethod
    def _extract_text_blocks(content: list) -> str:
        """Extract text from a list of content blocks."""
        return " ".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )[:2000]
