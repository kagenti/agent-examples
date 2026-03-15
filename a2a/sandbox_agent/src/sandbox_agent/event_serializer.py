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
import uuid
from abc import ABC, abstractmethod
from typing import Any

from sandbox_agent import plan_store as ps

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

    # Nodes whose events are sub-items of the preceding node visit
    # (they don't get their own node_visit number).
    _TOOL_NODES = frozenset({"tools", "planner_tools", "reflector_tools"})

    def __init__(self, loop_id: str | None = None, context_id: str | None = None) -> None:
        self._loop_id = loop_id or str(uuid.uuid4())[:8]
        self._step_index = 0
        self._event_counter = 0  # global sequence number for ordering
        self._node_visit = 0     # graph node visit counter (main sections)
        self._sub_index = 0      # position within current node visit
        self._last_node_key: str = ""  # track previous node for visit grouping
        self._micro_step: int = 0
        self._context_id = context_id or "unknown"
        self._last_call_id: str = ""
        self._prev_node: str | None = None  # previous node for node_transition events
        self._current_node: str = ""  # current LangGraph node name

    def serialize(self, key: str, value: dict) -> str:
        # Track current LangGraph node name for enrichment
        self._current_node = key

        # Emit node_transition meta-event when the node changes
        transition_line: str | None = None
        if self._prev_node is not None and key != self._prev_node:
            self._event_counter += 1
            transition_event = {
                "type": "node_transition",
                "loop_id": self._loop_id,
                "from_node": self._prev_node,
                "to_node": key,
                "event_index": self._event_counter,
                "langgraph_node": key,
            }
            transition_line = json.dumps(transition_event)
        self._prev_node = key

        # Node visit tracking:
        # - Tool nodes (tools, planner_tools, reflector_tools) inherit parent visit
        # - Same node type re-entering (executor→tools→executor) stays on same visit
        # - Different node type (executor→reflector, reflector→planner) = new visit
        if key not in self._TOOL_NODES:
            if key != self._last_node_key:
                self._node_visit += 1
                self._sub_index = 0
            self._last_node_key = key
        # event_counter incremented per JSON line in post-processing.

        # Track actual plan step from state for step grouping
        current_step = value.get("current_step")
        if current_step is not None:
            new_step = current_step + 1  # 1-based for display
            if new_step != self._step_index:
                self._step_index = new_step
                self._micro_step = 0  # reset micro_step on plan step change

        # Reasoning-loop nodes may emit state fields instead of messages
        if key == "router":
            # Router is an internal node — emit minimal event for logging
            route = value.get("_route", "new")
            result = json.dumps({
                "type": "router",
                "loop_id": self._loop_id,
                "route": route,
                "plan_status": value.get("plan_status", ""),
            })
        elif key == "planner":
            result = self._serialize_planner(value)
        elif key == "reflector":
            result = self._serialize_reflector(value)
        elif key == "step_selector":
            # Reset micro_step on every step transition
            self._micro_step = 0
            current_step = value.get("current_step", 0)
            plan_steps = value.get("plan_steps", [])
            step_desc = ""
            if current_step < len(plan_steps):
                step_entry = plan_steps[current_step]
                step_desc = step_entry.get("description", "") if isinstance(step_entry, dict) else str(step_entry)
            brief = value.get("skill_instructions", "")
            # Strip the "STEP BRIEF FROM COORDINATOR:" prefix
            if "STEP BRIEF" in brief:
                brief = brief.split("---")[0].replace("STEP BRIEF FROM COORDINATOR:", "").strip()
            result = json.dumps({
                "type": "step_selector",
                "loop_id": self._loop_id,
                "current_step": current_step,
                "description": f"Advancing to step {current_step + 1}: {step_desc[:80]}",
                "brief": brief[:500],
                "done": value.get("done", False),
            })
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

        # Append budget_update event if _budget_summary is in the value dict
        budget_summary = value.get("_budget_summary")
        if budget_summary and isinstance(budget_summary, dict):
            budget_event = json.dumps({
                "type": "budget_update",
                "loop_id": self._loop_id,
                **budget_summary,
            })
            result = result + "\n" + budget_event

        # Post-process: ensure ALL event lines have step + unique event_index.
        # Each JSON line gets its own event_index (no duplicates).
        # Legacy event types (plan, plan_step, reflection) are skipped from
        # indexing to avoid inflating the counter.
        enriched_lines = []

        # Prepend node_transition event if one was emitted
        if transition_line is not None:
            enriched_lines.append(transition_line)

        for line in result.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
                if "step" not in evt:
                    cs = evt.get("current_step")
                    evt["step"] = (cs + 1) if cs is not None else self._step_index
                event_type = evt.get("type", "?")
                self._event_counter += 1
                evt["event_index"] = self._event_counter
                evt["node_visit"] = self._node_visit
                evt["sub_index"] = self._sub_index
                evt["langgraph_node"] = self._current_node
                self._sub_index += 1
                enriched_lines.append(json.dumps(evt))
            except json.JSONDecodeError:
                enriched_lines.append(line)
                event_type = "parse_error"
            logger.info("SERIALIZE session=%s loop=%s type=%s step=%s ei=%s",
                self._context_id, self._loop_id, event_type,
                self._step_index, self._event_counter,
                extra={"session_id": self._context_id, "node": key,
                       "event_type": event_type, "step": self._step_index})

        return "\n".join(enriched_lines)

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

        # Emit sub_events: thinking iterations, tool calls, tool results
        sub_events = _v.get("_sub_events", [])
        for se in sub_events:
            se_type = se.get("type", "")
            if se_type == "thinking":
                thinking_event = {
                    "type": "thinking",
                    "loop_id": self._loop_id,
                    "cycle": se.get("cycle", 1),
                    "iteration": se.get("iteration", 1),
                    "total_iterations": se.get("total_iterations", 1),
                    "reasoning": se.get("reasoning", "")[:50000],
                    "node": se.get("node", "executor"),
                    "model": se.get("model", ""),
                    "prompt_tokens": se.get("prompt_tokens", 0),
                    "completion_tokens": se.get("completion_tokens", 0),
                }
                for field in ("_system_prompt", "_prompt_messages", "_bound_tools", "_llm_response"):
                    if field in se:
                        thinking_event[field.lstrip("_")] = se[field]
                parts.append(json.dumps(thinking_event))
            elif se_type == "tool_call":
                parts.append(json.dumps({
                    "type": "tool_call",
                    "loop_id": self._loop_id,
                    "call_id": se.get("call_id", ""),
                    "cycle": se.get("cycle", 1),
                    "tools": se.get("tools", []),
                }))
            elif se_type == "tool_result":
                parts.append(json.dumps({
                    "type": "tool_result",
                    "loop_id": self._loop_id,
                    "call_id": se.get("call_id", ""),
                    "cycle": se.get("cycle", 1),
                    "name": se.get("name", "unknown"),
                    "output": se.get("output", "")[:2000],
                    "status": se.get("status", "success"),
                }))

        self._micro_step += 1

        # Skip micro_reasoning for dedup responses (no LLM call happened)
        if not _v.get("_dedup"):
            # Annotate micro_reasoning with thinking count
            if sub_events:
                _v = {**_v, "_thinking_count": len(sub_events)}
            parts.append(self._serialize_micro_reasoning(msg, _v))

        plan = _v.get("plan", [])
        model = _v.get("model", "")
        prompt_tokens = _v.get("prompt_tokens", 0)
        completion_tokens = _v.get("completion_tokens", 0)
        prompt_data = self._extract_prompt_data(_v)

        # Emit executor_step event so UI shows which step is executing
        current_plan_step = _v.get("current_step", 0)
        step_payload = {
            "type": "executor_step",
            "loop_id": self._loop_id,
            "plan_step": current_plan_step,
            "iteration": _v.get("iteration", 0),
            "total_steps": len(plan) if plan else 0,
            "description": text[:200] if text else "",
            "reasoning": text[:2000] if text else "",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            **prompt_data,
        }
        parts.append(json.dumps(step_payload))

        if tool_calls:
            # Use LangGraph's tool_call_id for proper pairing with tool_result
            tc0 = tool_calls[0] if tool_calls else {}
            call_id = (
                tc0.get("id") if isinstance(tc0, dict)
                else getattr(tc0, "id", None)
            ) or str(uuid.uuid4())[:8]
            self._last_call_id = call_id
            parts.append(json.dumps({
                "type": "tool_call",
                "loop_id": self._loop_id,
                "call_id": call_id,
                "tools": [
                    _safe_tc(tc)
                    for tc in tool_calls
                ],
            }))
            return "\n".join(parts)

        # Emit tool_call event for text-parsed tools (no structured tool_calls)
        parsed_tools = _v.get("parsed_tools", [])
        if parsed_tools:
            call_id = str(uuid.uuid4())[:8]
            self._last_call_id = call_id
            parts.append(json.dumps({
                "type": "tool_call",
                "loop_id": self._loop_id,
                "call_id": call_id,
                "tools": [
                    {"name": t["name"], "args": t.get("args", {})}
                    for t in parsed_tools
                ],
            }))

        return "\n".join(parts)

    def _serialize_micro_reasoning(self, msg: Any, value: dict) -> str:
        """Emit a micro_reasoning event capturing the LLM's intermediate reasoning."""
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            text = self._extract_text_blocks(content)
        else:
            text = str(content)[:50000] if content else ""

        tool_calls = getattr(msg, "tool_calls", [])
        next_action = "tool_call" if tool_calls else "done"

        # When the LLM responds with only tool calls and no text reasoning,
        # generate a summary so the micro-reasoning block isn't empty.
        if not text and tool_calls:
            summaries = []
            for tc in tool_calls[:5]:
                name = tc.get("name", "?")
                args = tc.get("args", {})
                args_str = json.dumps(args, default=str)[:200]
                summaries.append(f"→ {name}({args_str})")
            text = "Decided next action:\n" + "\n".join(summaries)

        event: dict = {
            "type": "micro_reasoning",
            "loop_id": self._loop_id,
            "micro_step": self._micro_step,
            "after_call_id": self._last_call_id,
            "reasoning": text[:50000],
            "next_action": next_action,
            "model": value.get("model", ""),
            "prompt_tokens": value.get("prompt_tokens", 0),
            "completion_tokens": value.get("completion_tokens", 0),
            **self._extract_prompt_data(value),
        }
        # Include previous tool result for UI context (shows WHY this decision)
        prev = value.get("_last_tool_result")
        if prev:
            event["previous_tool"] = prev
        # Annotate with thinking iteration count for UI badge
        tc = value.get("_thinking_count", 0)
        if tc:
            event["thinking_count"] = tc
        return json.dumps(event)

    def _serialize_tool_result(self, msg: Any) -> str:
        """Serialize a tool node output with loop_id."""
        name = getattr(msg, "name", "unknown")
        content = getattr(msg, "content", "")
        content_str = str(content)
        # Determine error status from exit code, not content keywords.
        # The shell tool appends "EXIT_CODE: N" for non-zero exits.
        # Keyword matching (e.g. "failure", "error") causes false positives
        # when command output contains those words in normal data.
        import re as _re
        exit_match = _re.search(r"EXIT_CODE:\s*(\d+)", content_str)
        is_error = (
            (exit_match is not None and exit_match.group(1) != "0")
            or content_str.startswith("\u274c")
            or content_str.startswith("Error: ")
            or "Permission denied" in content_str
            or "command not found" in content_str
        )
        status = "error" if is_error else "success"
        # Use LangGraph's tool_call_id for proper pairing with tool_call
        call_id = getattr(msg, "tool_call_id", None) or self._last_call_id
        return json.dumps({
            "type": "tool_result",
            "loop_id": self._loop_id,
            "call_id": call_id,
            "name": str(name),
            "output": content_str[:2000],
            "status": status,
        })

    @staticmethod
    def _enrich_with_plan_store(payload: dict, value: dict) -> None:
        """Add PlanStore flat steps to payload if available."""
        store = value.get("_plan_store", {})
        if store and store.get("steps"):
            payload["plan_steps"] = ps.to_flat_plan_steps(store)

    @staticmethod
    def _extract_prompt_data(value: dict) -> dict:
        """Extract prompt visibility fields from node output."""
        data: dict = {}
        sp = value.get("_system_prompt", "")
        if sp:
            data["system_prompt"] = sp[:50000]
        pm = value.get("_prompt_messages")
        if pm:
            data["prompt_messages"] = pm[:100]  # max 100 messages
        bt = value.get("_bound_tools")
        if bt:
            data["bound_tools"] = bt[:50]  # max 50 tools
        lr = value.get("_llm_response")
        if lr:
            data["llm_response"] = lr
        return data

    def _serialize_planner(self, value: dict) -> str:
        """Serialize a planner node output — emits planner_output + legacy plan."""
        plan_steps = value.get("plan_steps", [])
        plan = [s.get("description", "") for s in plan_steps] if plan_steps else value.get("plan", [])
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
        prompt_data = self._extract_prompt_data(value)

        # Distinguish initial plan from replan
        is_replan = iteration > 1
        event_type = "replanner_output" if is_replan else "planner_output"

        payload = {
            "type": event_type,
            "loop_id": self._loop_id,
            "steps": plan,
            "iteration": iteration,
            "content": text,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            **prompt_data,
        }

        self._enrich_with_plan_store(payload, value)

        return json.dumps(payload)

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

        # Strip prompt echo from assessment — the LLM sometimes echoes the
        # system prompt instructions.  Extract only the actual decision word
        # or a brief justification, never the echoed prompt.
        assessment = text.strip()

        # If the response contains prompt markers, it's an echo — just use the decision.
        prompt_markers = (
            "Output the single word:",
            "output ONLY the decision word",
            "Decide ONE of the following",
            "DECISION PROCESS:",
            "STALL DETECTION:",
            "REPLAN RULES:",
        )
        is_prompt_echo = any(marker in assessment for marker in prompt_markers)
        if is_prompt_echo or not assessment or len(assessment) > 200:
            assessment = decision

        # Reset micro_step counter for next iteration
        self._micro_step = 0

        model = value.get("model", "")
        prompt_tokens = value.get("prompt_tokens", 0)
        completion_tokens = value.get("completion_tokens", 0)
        iteration = value.get("iteration", 0)
        prompt_data = self._extract_prompt_data(value)

        payload = {
            "type": "reflector_decision",
            "loop_id": self._loop_id,
            "decision": decision,
            "assessment": assessment,
            "iteration": iteration,
            "done": done,
            "current_step": current_step,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            **prompt_data,
        }

        self._enrich_with_plan_store(payload, value)

        return json.dumps(payload)

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
        prompt_data = self._extract_prompt_data(value)

        payload = {
            "type": "reporter_output",
            "loop_id": self._loop_id,
            "content": final_answer[:2000],
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            **prompt_data,
        }

        files_touched = value.get("files_touched", [])
        if files_touched:
            payload["files_touched"] = files_touched[:30]

        return json.dumps(payload)

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
