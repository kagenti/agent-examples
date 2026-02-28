"""Framework-specific event serializers for structured JSON streaming.

Each agent framework (LangGraph, CrewAI, AG2) has its own internal event
format. Serializers convert framework events into a common JSON schema
that the backend and frontend understand.

Event types:
    tool_call     — LLM decided to call one or more tools
    tool_result   — A tool returned output
    llm_response  — LLM generated text (no tool calls)
    error         — An error occurred during execution
    hitl_request  — Human-in-the-loop approval is needed
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


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
    """

    def serialize(self, key: str, value: dict) -> str:
        msgs = value.get("messages", [])
        if not msgs:
            return json.dumps({"type": "llm_response", "content": f"[{key}]"})

        msg = msgs[-1]

        if key == "assistant":
            return self._serialize_assistant(msg)
        elif key == "tools":
            return self._serialize_tool_result(msg)
        else:
            # Unknown node — treat as informational
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                text = self._extract_text_blocks(content)
            else:
                text = str(content)[:2000] if content else f"[{key}]"
            return json.dumps({"type": "llm_response", "content": text})

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
                    {
                        "name": tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown"),
                        "args": tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {}),
                    }
                    for tc in tool_calls
                ],
            }))
            return "\n".join(parts)

        return json.dumps({"type": "llm_response", "content": text})

    def _serialize_tool_result(self, msg: Any) -> str:
        """Serialize a tool node output."""
        name = getattr(msg, "name", "unknown")
        content = getattr(msg, "content", "")
        return json.dumps({
            "type": "tool_result",
            "name": str(name),
            "output": str(content)[:2000],
        })

    @staticmethod
    def _extract_text_blocks(content: list) -> str:
        """Extract text from a list of content blocks."""
        return " ".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )[:2000]
