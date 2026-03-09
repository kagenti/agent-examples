"""Plan-execute-reflect reasoning loop node functions.

Four LangGraph node functions implement structured multi-step reasoning:

1. **planner** — Decomposes the user request into numbered steps.
   Detects simple (single-step) requests and marks them done-after-execute.
2. **executor** — Runs the current plan step with bound tools (existing
   react pattern).
3. **reflector** — Reviews execution output, decides: ``continue`` (next
   step), ``replan``, ``done``, or ``hitl``.
4. **reporter** — Formats accumulated step results into a final answer.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from sandbox_agent.budget import AgentBudget

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text-based tool call parser
# ---------------------------------------------------------------------------
# Some model servers (e.g. vLLM without --enable-auto-tool-choice) return
# tool invocations as text like:
#   [shell(command="ls -la"), file_read(path="foo.py")]
# instead of structured tool_calls in the OpenAI response format.
# This parser converts that text into proper AIMessage.tool_calls so
# LangGraph's tools_condition routes to the ToolNode.
# ---------------------------------------------------------------------------

# Matches: tool_name(key="value", key2="value2")
# Handles: shell("ls") (positional), shell(command="ls") (keyword)
_TOOL_CALL_RE = re.compile(
    r'(\w+)\(([^)]*)\)',
)

# Matches Llama 4 Scout format: [label, tool_name]{"key": "value"}
# Examples: [clone_repo, shell]{"command": "git clone ..."}
#           [rca:ci, delegate]{"task": "analyze CI logs"}
_LABEL_TOOL_JSON_RE = re.compile(
    r'\[[^\]]*,\s*(\w+)\]\s*(\{[^}]+\})',
)

# Known tool names — only parse calls for tools we actually have
_KNOWN_TOOLS = {"shell", "file_read", "file_write", "grep", "glob", "web_fetch", "explore", "delegate"}

# First-param defaults for tools that accept a positional argument
_POSITIONAL_PARAM = {
    "shell": "command",
    "file_read": "path",
    "grep": "pattern",
    "glob": "pattern",
    "web_fetch": "url",
    "explore": "query",
    "delegate": "task",
}


def _parse_kwargs(args_str: str, tool_name: str) -> dict[str, Any]:
    """Parse 'key="value", key2="value2"' or '"positional"' into a dict."""
    args_str = args_str.strip()
    if not args_str:
        return {}

    result: dict[str, Any] = {}

    # Try keyword arguments first: key="value" or key='value'
    kw_pattern = re.compile(r'(\w+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\')')
    kw_matches = kw_pattern.findall(args_str)
    if kw_matches:
        for key, val_dq, val_sq in kw_matches:
            val = val_dq if val_dq else val_sq
            val = val.replace('\\"', '"').replace("\\'", "'")
            result[key] = val
        return result

    # Positional: just a quoted string like "ls -la" or 'ls -la'
    pos_match = re.match(r'^["\'](.+?)["\']$', args_str, re.DOTALL)
    if pos_match:
        param_name = _POSITIONAL_PARAM.get(tool_name, "input")
        result[param_name] = pos_match.group(1).replace('\\"', '"')
        return result

    # Unquoted positional (rare but handle it)
    param_name = _POSITIONAL_PARAM.get(tool_name, "input")
    result[param_name] = args_str
    return result


def parse_text_tool_calls(content: str) -> list[dict[str, Any]]:
    """Extract tool calls from text content.

    Returns a list of dicts matching LangChain ToolCall format:
      [{"name": "shell", "args": {"command": "ls"}, "id": "...", "type": "tool_call"}]

    Returns empty list if no recognizable tool calls found.
    """
    if not content:
        return []

    # Look for the pattern: [tool(...), tool(...)] or just tool(...)
    # Strip surrounding brackets if present
    text = content.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
        # Remove trailing comma
        if text.endswith(","):
            text = text[:-1].strip()

    calls = []

    # Try Llama 4 format first: [label, tool_name]{"key": "value"}
    for match in _LABEL_TOOL_JSON_RE.finditer(content):
        tool_name = match.group(1)
        json_str = match.group(2)
        if tool_name not in _KNOWN_TOOLS:
            continue
        try:
            args = json.loads(json_str)
            if isinstance(args, dict):
                calls.append({
                    "name": tool_name,
                    "args": args,
                    "id": f"text-{uuid.uuid4().hex[:12]}",
                    "type": "tool_call",
                })
        except json.JSONDecodeError:
            continue

    if calls:
        return calls

    # Slash-command format: /shell\ncommand or /file_read\npath
    slash_re = re.compile(r'^/(' + '|'.join(_KNOWN_TOOLS) + r')\s*\n(.+)', re.DOTALL)
    slash_match = slash_re.match(text)
    if slash_match:
        tool_name = slash_match.group(1)
        arg_text = slash_match.group(2).strip()
        arg_key = {"shell": "command", "file_read": "path", "file_write": "path",
                    "grep": "pattern", "glob": "pattern", "web_fetch": "url"}.get(tool_name, "command")
        calls.append({
            "name": tool_name,
            "args": {arg_key: arg_text.split("\n")[0].strip()},
            "id": f"text-{uuid.uuid4().hex[:12]}",
            "type": "tool_call",
        })
        return calls

    # Bash code block: ```bash\ncommand\n``` or ```sh\ncommand\n```
    bash_re = re.compile(r'```(?:bash|sh)\s*\n(.+?)\n```', re.DOTALL)
    bash_match = bash_re.search(text)
    if bash_match:
        cmd = bash_match.group(1).strip()
        if cmd and len(cmd) < 2000:
            calls.append({
                "name": "shell",
                "args": {"command": cmd},
                "id": f"text-{uuid.uuid4().hex[:12]}",
                "type": "tool_call",
            })
            return calls

    # Fall back to legacy format: tool_name(args)
    for match in _TOOL_CALL_RE.finditer(text):
        tool_name = match.group(1)
        args_str = match.group(2)

        if tool_name not in _KNOWN_TOOLS:
            continue

        args = _parse_kwargs(args_str, tool_name)
        calls.append({
            "name": tool_name,
            "args": args,
            "id": f"text-{uuid.uuid4().hex[:12]}",
            "type": "tool_call",
        })

    return calls


def maybe_patch_tool_calls(response: AIMessage) -> AIMessage:
    """If the response has no tool_calls but contains text-based calls, patch them in."""
    if response.tool_calls:
        # Model returned structured tool_calls — use as-is
        return response

    content = response.content
    if isinstance(content, list):
        # Multi-part content — extract text parts
        content = " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )

    parsed = parse_text_tool_calls(content)
    if not parsed:
        return response

    logger.info(
        "Parsed %d text-based tool call(s): %s",
        len(parsed),
        [c["name"] for c in parsed],
    )

    # Create a new AIMessage with the parsed tool_calls
    return AIMessage(
        content="",  # Clear text content — tools will produce output
        tool_calls=parsed,
    )

# Default budget — used when no explicit budget is passed.
DEFAULT_BUDGET = AgentBudget()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
You are a planning module for a sandboxed coding assistant.

Given the user's request and any prior execution results, produce a concise
numbered plan.  Each step should be a single actionable item that can be
executed with the available tools (shell, file_read, file_write, grep, glob,
web_fetch, explore, delegate).

Rules:
- If the request needs NO tools (just a text answer, saying something,
  answering a question from memory, or repeating text), output EXACTLY:
  1. Respond to the user.
  DO NOT add extra steps for thinking, analyzing, or verifying.
- If the request is a single command or a trivial file operation,
  output EXACTLY one step.
- NEVER create multi-step plans for simple requests. One command = one step.
- Keep steps concrete and tool-oriented — no vague "analyze" or "think" steps.
- For multi-step analysis, debugging, or investigation tasks, add a final
  step: "Write findings summary to report.md" with sections: Problem,
  Investigation, Root Cause, Resolution.
- For complex investigations that can be parallelized, use the **delegate**
  tool to spawn child agent sessions for independent research tasks. Each
  child session runs in its own workspace and reports back results.
- Number each step starting at 1.
- Output ONLY the numbered list, nothing else.

Example for a text-only request ("Say exactly: hello world"):
1. Respond to the user.

Example for a question ("What was the marker text?"):
1. Respond to the user.

Example for a simple request ("list files"):
1. Run `ls -la` in the workspace.

Example for a single command ("run echo test"):
1. Run `echo test` in the shell.

Example for a complex request ("create a Python project with tests"):
1. Create the directory structure with `mkdir -p src tests`.
2. Write `src/main.py` with the main module code.
3. Write `tests/test_main.py` with pytest tests.
4. Run `python -m pytest tests/` to verify tests pass.

Example for an RCA/CI investigation ("analyze CI failures for owner/repo PR #758"):
1. Clone and set up remotes: `git clone https://github.com/owner/repo.git repos/repo && cd repos/repo && git remote set-url origin https://github.com/owner/repo.git`.
2. From the repo dir, list failures: `cd repos/repo && gh run list --status failure --limit 5`.
3. Download failure logs: `cd repos/repo && gh run view <run_id> --log-failed > ../../output/ci-run.log`.
4. Extract errors: `grep -C 5 'FAILED\|ERROR\|AssertionError' output/ci-run.log`.
5. Write findings to report.md with sections: Root Cause, Impact, Fix.

IMPORTANT for gh CLI:
- Always clone the target repo FIRST into repos/, then `cd repos/<name>` before gh commands.
- Set origin to the UPSTREAM repo URL (not a fork) so gh resolves the correct repo.
- gh auto-detects the repo from git remote "origin" — it MUST run inside the cloned repo.
- Use `cd repos/<name> && gh <command>` in a single shell call (each call starts from workspace root).
- Save output to output/ for later analysis.
"""

_EXECUTOR_SYSTEM = """\
You are a sandboxed coding assistant executing step {current_step} of a plan.

Current step: {step_text}

Available tools:
- **shell**: Execute a shell command. Returns stdout+stderr and exit code.
- **file_read**: Read a file from the workspace.
- **file_write**: Write content to a file in the workspace.
- **grep**: Search file contents with regex. Faster than shell grep, workspace-scoped.
- **glob**: Find files by pattern (e.g. '**/*.py'). Faster than shell find.
- **web_fetch**: Fetch content from a URL (allowed domains only).
- **explore**: Spawn a read-only sub-agent for codebase research.
- **delegate**: Spawn a child agent session for a delegated task.

CRITICAL RULES:
- You MUST use the function/tool calling API to execute actions.
- DO NOT write tool names as text (like "/shell", "shell(...)", or code blocks).
  These are NOT how you call tools. Use the function calling API instead.
- DO NOT write or invent command output. Call the tool, wait for the result.
- If a tool call fails, report the ACTUAL error — do not invent output.
- Call ONE tool at a time. Wait for the result before the next call.
- Slash commands like /rca:ci are for humans, not for you. You use tools.

Execute ONLY this step. You MUST make at least one tool call.
When done, summarize what you accomplished with the actual tool output.
"""

_REFLECTOR_SYSTEM = """\
You are a reflection module reviewing the output of a plan step.

Plan:
{plan_text}

Current step ({current_step}): {step_text}
Step result: {step_result}

Decide ONE of the following (output ONLY the decision word):
- **continue** — Step succeeded; move to the next step.
- **replan** — Step failed or revealed new information; re-plan remaining work.
- **done** — All steps are complete or the task is fully answered.
- **hitl** — Human input is needed to proceed.

Output the single word: continue, replan, done, or hitl.
"""

_REPORTER_SYSTEM = """\
You are a reporting module.  Summarize the results of all executed steps
into a clear, concise final answer for the user.

Plan:
{plan_text}

Step results:
{results_text}

RULES:
- Only report facts from actual tool output — NEVER fabricate data.
- If a step failed or returned an error, include the error in the report.
- If no real data was obtained, say "Unable to retrieve data" rather than
  making up results.
- Include relevant command output, file paths, or next steps.
- Do NOT include the plan itself — just the results.
"""


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def _is_trivial_text_request(messages: list) -> bool:
    """Detect requests that need no tools — just a text response.

    Matches patterns like "Say exactly: ...", "What was the marker?",
    simple greetings, or questions that can be answered from conversation
    context alone.
    """
    if not messages:
        return False
    last = messages[-1]
    content = getattr(last, "content", "")
    if isinstance(content, list):
        content = " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    text = str(content).strip().lower()
    if not text:
        return False

    # Patterns that clearly need no tools
    trivial_patterns = (
        "say exactly",
        "repeat ",
        "what was the marker",
        "what did i say",
        "what did i tell",
        "hello",
        "hi",
        "who are you",
    )
    return any(text.startswith(p) or p in text for p in trivial_patterns)


async def planner_node(
    state: dict[str, Any],
    llm: Any,
) -> dict[str, Any]:
    """Decompose the user request into a numbered plan.

    On re-entry (iteration > 0), the planner also sees prior step results so
    it can adjust the remaining plan.
    """
    messages = state["messages"]
    iteration = state.get("iteration", 0)
    step_results = state.get("step_results", [])

    # Fast-path: trivial text-only requests skip the planner LLM call entirely
    if iteration == 0 and _is_trivial_text_request(messages):
        logger.info("Fast-path: trivial text request — single-step plan, no LLM call")
        return {
            "plan": ["Respond to the user."],
            "current_step": 0,
            "iteration": 1,
            "done": False,
        }

    # Build context for the planner
    context_parts = []
    if iteration > 0 and step_results:
        context_parts.append("Previous step results:")
        for i, result in enumerate(step_results, 1):
            context_parts.append(f"  Step {i}: {result}")
        context_parts.append("")
        context_parts.append("Adjust the plan for remaining work.")

    system_content = _PLANNER_SYSTEM
    if context_parts:
        system_content += "\n" + "\n".join(context_parts)

    # Prepend skill instructions when a skill was loaded from metadata.
    skill_instructions = state.get("skill_instructions", "")
    if skill_instructions:
        system_content = skill_instructions + "\n\n" + system_content

    plan_messages = [SystemMessage(content=system_content)] + messages
    response = await llm.ainvoke(plan_messages)

    # Extract token usage from the LLM response
    usage = getattr(response, 'usage_metadata', None) or {}
    prompt_tokens = usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)

    # Parse numbered steps from the response
    plan = _parse_plan(response.content)

    logger.info("Planner produced %d steps (iteration %d): %s", len(plan), iteration, plan)

    return {
        "messages": [response],
        "plan": plan,
        "current_step": 0,
        "iteration": iteration + 1,
        "done": False,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


async def executor_node(
    state: dict[str, Any],
    llm_with_tools: Any,
) -> dict[str, Any]:
    """Execute the current plan step using the LLM with bound tools."""
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if current_step >= len(plan):
        # No more steps — signal completion to reflector
        return {
            "messages": [AIMessage(content="All plan steps completed.")],
            "done": True,
        }

    step_text = plan[current_step]
    system_content = _EXECUTOR_SYSTEM.format(
        current_step=current_step + 1,
        step_text=step_text,
    )

    # Prepend skill instructions when a skill was loaded from metadata.
    skill_instructions = state.get("skill_instructions", "")
    if skill_instructions:
        system_content = skill_instructions + "\n\n" + system_content

    # Include the conversation history so the executor has full context
    messages = [SystemMessage(content=system_content)] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)

    # Extract token usage from the LLM response
    usage = getattr(response, 'usage_metadata', None) or {}
    prompt_tokens = usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)

    # If the model returned text-based tool calls instead of structured
    # tool_calls (common with vLLM without --enable-auto-tool-choice),
    # parse them so tools_condition routes to the ToolNode.
    # Capture the pre-patch content for event serialization.
    pre_patch_content = response.content
    had_structured_tools = bool(response.tool_calls)
    response = maybe_patch_tool_calls(response)

    # -- Dedup: skip tool calls that already have ToolMessage responses ------
    # The text-based parser generates fresh UUIDs each invocation, so
    # LangGraph treats re-parsed calls as new work.  Match on (name, args)
    # against already-executed calls in the message history to break the
    # executor→tools→executor loop.
    if response.tool_calls:
        executed: set[tuple[str, str]] = set()
        messages = state.get("messages", [])
        # Build a map from tool_call_id → (name, args) for all AIMessage
        # tool calls, then record those that have a ToolMessage response.
        tc_id_to_key: dict[str, tuple[str, str]] = {}
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    key = (tc["name"], repr(sorted(tc["args"].items())))
                    tc_id_to_key[tc["id"]] = key
            elif isinstance(msg, ToolMessage):
                key = tc_id_to_key.get(msg.tool_call_id)
                if key is not None:
                    executed.add(key)

        new_calls = [
            tc for tc in response.tool_calls
            if (tc["name"], repr(sorted(tc["args"].items()))) not in executed
        ]

        if len(new_calls) < len(response.tool_calls):
            skipped = len(response.tool_calls) - len(new_calls)
            logger.info(
                "Dedup: skipped %d already-executed tool call(s)", skipped,
            )
            if not new_calls:
                # All calls already executed — return text so tools_condition
                # routes to reflector instead of looping back to tools.
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "All tool calls for this step have already "
                                "been executed. Proceeding to review results."
                            ),
                        )
                    ]
                }
            # Keep only genuinely new calls
            response = AIMessage(
                content=response.content,
                tool_calls=new_calls,
            )

    # Build parsed_tools list for event serialization when tools came
    # from text parsing (not structured tool_calls).
    parsed_tools: list[dict[str, Any]] = []
    if not had_structured_tools and response.tool_calls:
        parsed_tools = [
            {"name": tc["name"], "args": tc.get("args", {})}
            for tc in response.tool_calls
        ]

    result: dict[str, Any] = {
        "messages": [response],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if parsed_tools:
        result["parsed_tools"] = parsed_tools
    return result


async def reflector_node(
    state: dict[str, Any],
    llm: Any,
    budget: AgentBudget | None = None,
) -> dict[str, Any]:
    """Review step output and decide whether to continue, replan, or finish.

    Parameters
    ----------
    budget:
        Optional :class:`AgentBudget` for enforcing iteration limits.
        When the budget is exceeded the reflector forces ``done``.
    """
    if budget is None:
        budget = DEFAULT_BUDGET

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    step_results = list(state.get("step_results", []))
    iteration = state.get("iteration", 0)
    done = state.get("done", False)

    # If executor signaled done (ran out of steps), go straight to done
    if done:
        return {"done": True}

    # Budget guard — force termination if iterations exceeded
    if iteration >= budget.max_iterations:
        logger.warning(
            "Budget exceeded: %d/%d iterations used — forcing done",
            iteration, budget.max_iterations,
        )
        return {
            "step_results": step_results,
            "current_step": current_step + 1,
            "done": True,
        }

    # Extract the result from the last message
    messages = state["messages"]
    last_content = ""
    if messages:
        last_msg = messages[-1]
        content = getattr(last_msg, "content", "")
        if isinstance(content, list):
            last_content = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            last_content = str(content)

    step_results.append(last_content[:500])

    step_text = plan[current_step] if current_step < len(plan) else "N/A"
    plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
    results_text = last_content[:1000]

    # Ask LLM to reflect
    system_content = _REFLECTOR_SYSTEM.format(
        plan_text=plan_text,
        current_step=current_step + 1,
        step_text=step_text,
        step_result=results_text,
    )
    reflect_messages = [SystemMessage(content=system_content)]
    response = await llm.ainvoke(reflect_messages)

    # Extract token usage from the LLM response
    usage = getattr(response, 'usage_metadata', None) or {}
    prompt_tokens = usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)

    decision = _parse_decision(response.content)
    logger.info("Reflector decision: %s (step %d/%d)", decision, current_step + 1, len(plan))

    if decision == "done" or current_step + 1 >= len(plan):
        return {
            "messages": [response],
            "step_results": step_results,
            "current_step": current_step + 1,
            "done": True,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    elif decision == "replan":
        # Feed back to planner — keep step_results, reset current_step
        return {
            "messages": [response],
            "step_results": step_results,
            "done": False,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    else:
        # continue — advance to next step
        return {
            "messages": [response],
            "step_results": step_results,
            "current_step": current_step + 1,
            "done": False,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }


async def reporter_node(
    state: dict[str, Any],
    llm: Any,
) -> dict[str, Any]:
    """Format accumulated step results into a final answer."""
    plan = state.get("plan", [])
    step_results = state.get("step_results", [])

    # For single-step plans, just pass through the last message
    if len(plan) <= 1:
        messages = state["messages"]
        if messages:
            last = messages[-1]
            content = getattr(last, "content", "")
            if isinstance(content, list):
                text = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                text = str(content)
            # Guard: if text is a bare reflector decision keyword
            # (e.g. budget exhaustion forces done with "continue"),
            # fall through to LLM-based summary from step_results.
            if not _BARE_DECISION_RE.match(text.strip()):
                return {"final_answer": text}
            # Fall through to LLM-based summary below
        elif not step_results:
            return {"final_answer": "No response generated."}

    plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
    results_text = "\n".join(
        f"Step {i+1}: {r}" for i, r in enumerate(step_results)
    )

    system_content = _REPORTER_SYSTEM.format(
        plan_text=plan_text,
        results_text=results_text,
    )
    messages = [SystemMessage(content=system_content)] + state["messages"]
    response = await llm.ainvoke(messages)

    # Extract token usage from the LLM response
    usage = getattr(response, 'usage_metadata', None) or {}
    prompt_tokens = usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)

    content = response.content
    if isinstance(content, list):
        text = " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        text = str(content)

    return {
        "messages": [response],
        "final_answer": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


# ---------------------------------------------------------------------------
# Routing function for reflector conditional edges
# ---------------------------------------------------------------------------


def route_reflector(state: dict[str, Any]) -> str:
    """Route from reflector: ``done`` → reporter, otherwise → planner."""
    if state.get("done", False):
        return "done"
    return "continue"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_plan(content: str | list) -> list[str]:
    """Extract numbered steps from LLM output.

    Accepts both plain strings and content-block lists (tool-calling models).
    Returns a list of step descriptions.
    """
    if isinstance(content, list):
        text = " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        text = str(content)

    steps: list[str] = []
    for line in text.strip().splitlines():
        line = line.strip()
        # Match lines starting with a number followed by . or )
        if line and len(line) > 2 and line[0].isdigit():
            # Strip the number prefix: "1. Do X" -> "Do X"
            for i, ch in enumerate(line):
                if ch in ".)" and i < 4:
                    step = line[i + 1:].strip()
                    if step:
                        steps.append(step)
                    break

    # Fallback: if parsing fails, treat the whole response as a single step
    if not steps:
        steps = [text.strip()[:500]]

    return steps


def _parse_decision(content: str | list) -> str:
    """Extract the reflector decision from LLM output.

    Returns one of: ``continue``, ``replan``, ``done``, ``hitl``.
    Defaults to ``continue`` if the output is ambiguous.
    """
    if isinstance(content, list):
        text = " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        text = str(content)

    text_lower = text.strip().lower()

    for decision in ("done", "replan", "hitl", "continue"):
        if decision in text_lower:
            return decision

    return "continue"


_BARE_DECISION_RE = re.compile(r'^(continue|replan|done|hitl)\s*$', re.IGNORECASE)
