"""Plan-execute-reflect reasoning loop node functions.

Five LangGraph node functions implement structured multi-step reasoning:

1. **router** — Entry point. Checks plan_status to decide: resume existing
   plan, replan with new context, or start fresh.
2. **planner** — Decomposes the user request into numbered steps.
   Detects simple (single-step) requests and marks them done-after-execute.
3. **executor** — Runs the current plan step with bound tools (existing
   react pattern).
4. **reflector** — Reviews execution output, decides: ``continue`` (next
   step), ``replan``, ``done``, or ``hitl``.  Updates per-step status.
5. **reporter** — Formats accumulated step results into a final answer.
   Sets terminal ``plan_status`` based on how the loop ended.

Plan state persists across A2A turns via the LangGraph checkpointer.
When the user or looper sends "continue", the router resumes execution
at the current step. Any other message triggers a replan that sees the
previous plan's progress.

# TODO: Research explicit PlanStore approach as alternative to checkpointer.
# Pros of PlanStore: plan queryable outside graph (UI), full schema control,
#   plan versioning independent of LangGraph internals.
# Cons: more code, risk of plan/checkpointer state divergence, need custom
#   persistence layer.  Current approach (A) uses checkpointer for atomic
#   state which is simpler and less error-prone.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from sandbox_agent.budget import AgentBudget

logger = logging.getLogger(__name__)

# Sentinel text returned by the executor when all tool calls in a step have
# already been executed (dedup logic).  This is an internal coordination
# message and must never appear in user-visible output.
_DEDUP_SENTINEL = (
    "Step completed — all requested tool calls "
    "have been executed and results are available."
)

import os as _os

# Messages that trigger plan resumption rather than replanning.
_CONTINUE_PHRASES = frozenset({
    "continue", "continue on the plan", "go on", "proceed",
    "keep going", "next", "carry on",
})


# ---------------------------------------------------------------------------
# PlanStep — structured per-step tracking
# ---------------------------------------------------------------------------


class PlanStep(TypedDict, total=False):
    """A single step in the plan with status tracking."""
    index: int
    description: str
    status: str          # "pending" | "running" | "done" | "failed" | "skipped"
    tool_calls: list[str]
    result_summary: str
    iteration_added: int


def _summarize_messages(messages: list) -> list[dict[str, str]]:
    """Summarize a message list for prompt visibility in the UI.

    Returns a list of {role, content_preview} dicts showing what
    was sent to the LLM.
    """
    result = []
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        text = str(content)
        # Tool calls
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            tc_names = [tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?") for tc in tool_calls]
            text = f"[tool_calls: {', '.join(tc_names)}] {text[:200]}"
        # ToolMessage
        tool_name = getattr(msg, "name", None)
        if role == "tool" and tool_name:
            text = f"[{tool_name}] {text[:300]}"
        else:
            text = text[:500]
        result.append({"role": role, "preview": text})
    return result


def _make_plan_steps(
    descriptions: list[str], iteration: int = 0
) -> list[PlanStep]:
    """Convert a list of step descriptions into PlanStep dicts."""
    return [
        PlanStep(
            index=i,
            description=desc,
            status="pending",
            tool_calls=[],
            result_summary="",
            iteration_added=iteration,
        )
        for i, desc in enumerate(descriptions)
    ]


def _plan_descriptions(plan_steps: list[PlanStep]) -> list[str]:
    """Extract flat description list from plan_steps (for backward compat)."""
    return [s.get("description", "") for s in plan_steps]


def _safe_format(template: str, **kwargs: Any) -> str:
    """Format a prompt template, falling back to raw template on errors."""
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError) as exc:
        logger.warning("Prompt format error (%s), using raw template", exc)
        return template


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

IMPORTANT: Almost every request requires tools. The user is asking you to DO
things, not just talk. Create file = file_write. Run command = shell.
Clone repo = shell. Read file = file_read. Search code = grep/glob.

Rules:
- Every step should name the specific tool to use.
- Keep steps concrete and tool-oriented — no vague "analyze" or "think" steps.
- For multi-step analysis, debugging, or investigation tasks, add a final
  step: "Write findings summary to report.md" with sections: Problem,
  Investigation, Root Cause, Resolution.
- For complex investigations that can be parallelized, use the **delegate**
  tool to spawn child agent sessions for independent research tasks.
- Number each step starting at 1.
- Output ONLY the numbered list, nothing else.

Example ("create a file hello.txt with 'hello world'"):
1. Use file_write to create /workspace/hello.txt with content "hello world".

Example ("list files"):
1. Run `ls -la` in the workspace using shell.

Example ("create a Python project with tests"):
1. Create directory structure: shell(`mkdir -p src tests`).
2. Write src/main.py using file_write.
3. Write tests/test_main.py using file_write.
4. Run tests: shell(`python -m pytest tests/`).

Example ("analyze CI failures for owner/repo PR #758"):
1. Clone repo: shell(`git clone https://github.com/owner/repo.git repos/repo`).
2. List failures: shell(`cd repos/repo && gh run list --status failure --limit 5`).
3. Download logs: shell(`cd repos/repo && gh run view <run_id> --log-failed > ../../output/ci-run.log`).
4. Extract errors: grep(`FAILED|ERROR|AssertionError` in output/ci-run.log).
5. Write findings to report.md with sections: Root Cause, Impact, Fix.

IMPORTANT for gh CLI:
- GH_TOKEN and GITHUB_TOKEN are ALREADY set in the environment. Do NOT
  run `export GH_TOKEN=...` — it's unnecessary and will break auth.
- Always clone the target repo FIRST into repos/, then `cd repos/<name>` before gh commands.
- gh auto-detects the repo from git remote "origin" — it MUST run inside the cloned repo.
- Use `cd repos/<name> && gh <command>` in a single shell call (each call starts from workspace root).
- Save output to output/ for later analysis.
"""

_EXECUTOR_SYSTEM = """\
You are a sandboxed coding assistant executing step {current_step} of a plan.

Current step: {step_text}
Tool calls so far this step: {tool_call_count}/{max_tool_calls}

Available tools:
- **shell**: Execute a shell command. Returns stdout+stderr and exit code.
- **file_read**: Read a file from the workspace.
- **file_write**: Write content to a file in the workspace.
- **grep**: Search file contents with regex. Faster than shell grep, workspace-scoped.
- **glob**: Find files by pattern (e.g. '**/*.py'). Faster than shell find.
- **web_fetch**: Fetch content from a URL (allowed domains only).
- **explore**: Spawn a read-only sub-agent for codebase research.
- **delegate**: Spawn a child agent session for a delegated task.

EXECUTION MODEL — step-by-step with micro-reflection:
You operate in a loop: call ONE tool → see the result → decide what to do next.
After each tool result, THINK about what happened before calling the next tool.
- Did the command succeed? Check the exit code and output.
- If it failed, adapt your approach — don't blindly retry the same thing.
- If it succeeded, what's the logical next action for this step?

CRITICAL RULES:
- Call exactly ONE tool per response. You will see the result and can call another.
- You MUST use the function/tool calling API — not text descriptions of calls.
- DO NOT write or invent command output. Call the tool, wait for the result.
- If a tool call fails, report the ACTUAL error — do not invent output.
- Slash commands like /rca:ci are for humans, not for you. You use tools.
- If you cannot call a tool for any reason, respond with exactly:
  CANNOT_CALL_TOOL: <reason>

When the step is COMPLETE (goal achieved or cannot be achieved), stop calling
tools and summarize what you accomplished with the actual tool output.
"""

_REFLECTOR_SYSTEM = """\
You are a reflection module reviewing the output of a plan step.

Plan:
{plan_text}

Current step ({current_step}): {step_text}
Step result: {step_result}

Iteration: {iteration} of {max_iterations}
Replan count so far: {replan_count} (higher counts mean more rework — weigh this when deciding)
Tool calls this iteration: {tool_calls_this_iter}
Recent decisions: {recent_decisions}
{replan_history}

STALL DETECTION:
- If the executor made 0 tool calls, the step likely FAILED.
- If the step result is just text describing what WOULD be done (not actual
  tool output), that means the executor did not call any tools. Treat as failure.

REPLAN RULES:
- Do NOT replan with the same approach that already failed. If prior replans
  failed for the same reason, choose "done" instead.
- Each replan should try a fundamentally different strategy, not repeat the same steps.
- A high replan count suggests diminishing returns — consider "done" with
  partial results if you have already tried multiple distinct approaches.

Decide ONE of the following (output ONLY the decision word):
- **continue** — Step succeeded with real tool output; move to the next step.
- **replan** — Step failed or revealed new information; re-plan remaining work.
  (Only if you have a genuinely NEW approach to try.)
- **done** — All steps are complete, task is answered, OR agent is stuck.
- **hitl** — Human input is needed to proceed.

Output the single word: continue, replan, done, or hitl.
"""

_REPORTER_SYSTEM = """\
You are a reporting module.  Summarize the results of all executed steps
into a clear, concise final answer for the user.

Plan:
{plan_text}

Step status:
{step_status_text}

Step results:
{results_text}

RULES:
- Only report facts from actual tool output — NEVER fabricate data.
- If a step FAILED, explain WHY it failed (include the error message).
- If no real data was obtained, say "Unable to retrieve data" rather than
  making up results.
- Include relevant command output, file paths, or next steps.
- Do NOT include the plan itself — just the results.
"""


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


async def router_node(state: dict[str, Any]) -> dict[str, Any]:
    """Entry-point node: decide whether to resume, replan, or start fresh.

    Returns state updates that downstream conditional edges read via
    :func:`route_entry`.
    """
    plan_status = state.get("plan_status", "")
    plan_steps = state.get("plan_steps", [])
    messages = state.get("messages", [])

    # Extract the latest user message text
    last_text = ""
    if messages:
        content = getattr(messages[-1], "content", "")
        if isinstance(content, list):
            last_text = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            last_text = str(content)
    last_text_lower = last_text.strip().lower()

    has_active_plan = plan_status == "awaiting_continue" and len(plan_steps) > 0
    is_continue = last_text_lower in _CONTINUE_PHRASES

    if has_active_plan and is_continue:
        # Resume: mark next pending step as running
        current_step = state.get("current_step", 0)
        if current_step < len(plan_steps):
            plan_steps = list(plan_steps)  # copy for mutation
            plan_steps[current_step] = {**plan_steps[current_step], "status": "running"}
        logger.info(
            "Router: RESUME plan at step %d/%d (plan_status=%s)",
            current_step + 1, len(plan_steps), plan_status,
        )
        return {
            "_route": "resume",
            "plan_steps": plan_steps,
            "plan_status": "executing",
        }
    elif has_active_plan:
        # Replan: new instruction arrives while plan exists
        # Reset replan_count — this is a user-driven replan, not an agent loop
        logger.info(
            "Router: REPLAN — new message while plan active (plan_status=%s, steps=%d)",
            plan_status, len(plan_steps),
        )
        return {
            "_route": "replan",
            "plan_status": "executing",
            "original_request": last_text,
            "replan_count": 0,
            "recent_decisions": [],
        }
    else:
        # New: no active plan
        logger.info("Router: NEW plan (plan_status=%s)", plan_status)
        return {
            "_route": "new",
            "plan_status": "executing",
            "original_request": last_text,
        }


def route_entry(state: dict[str, Any]) -> str:
    """Conditional edge from router: resume → executor, else → planner."""
    route = state.get("_route", "new")
    if route == "resume":
        return "resume"
    return "plan"  # both "replan" and "new" go to planner


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

    prev_plan_steps = state.get("plan_steps", [])

    # Fast-path: trivial text-only requests skip the planner LLM call entirely
    if iteration == 0 and not prev_plan_steps and _is_trivial_text_request(messages):
        logger.info("Fast-path: trivial text request — single-step plan, no LLM call")
        trivial_steps = _make_plan_steps(["Respond to the user."], iteration=0)
        return {
            "plan": ["Respond to the user."],
            "plan_steps": trivial_steps,
            "plan_version": 1,
            "current_step": 0,
            "iteration": 1,
            "done": False,
        }

    # Build context for the planner — include previous plan with per-step status
    context_parts = []
    if prev_plan_steps:
        # Show the structured plan with per-step status
        context_parts.append("Previous plan (with status):")
        for ps in prev_plan_steps:
            idx = ps.get("index", 0)
            desc = ps.get("description", "")
            status = ps.get("status", "pending").upper()
            result = ps.get("result_summary", "")
            line = f"  {idx+1}. [{status}] {desc}"
            if result:
                line += f" — {result[:150]}"
            context_parts.append(line)
        done_count = sum(1 for s in prev_plan_steps if s.get("status") == "done")
        context_parts.append(f"Progress: {done_count}/{len(prev_plan_steps)} steps completed.")
        context_parts.append("")
    elif iteration > 0:
        # Fallback: use flat plan list for backward compat
        original_plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        if original_plan:
            context_parts.append("Original plan:")
            for i, step in enumerate(original_plan):
                status = "DONE" if i < current_step else "PENDING"
                context_parts.append(f"  {i+1}. [{status}] {step}")
            context_parts.append(f"Progress: {current_step}/{len(original_plan)} steps completed.")
            context_parts.append("")

    if iteration > 0 or prev_plan_steps:
        # Extract tool call history from messages
        tool_history = []
        for msg in messages:
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
                    args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    args_str = str(args)[:100]
                    tool_history.append(f"  CALLED: {name}({args_str})")
            if hasattr(msg, "name") and hasattr(msg, "content") and getattr(msg, "type", "") == "tool":
                output = str(getattr(msg, "content", ""))[:200]
                tool_history.append(f"  RESULT ({msg.name}): {output}")

        if tool_history:
            context_parts.append("Tool calls already executed (DO NOT repeat these):")
            context_parts.extend(tool_history[-20:])
            context_parts.append("")

        if step_results:
            context_parts.append("Previous step results:")
            for i, result in enumerate(step_results, 1):
                context_parts.append(f"  Step {i}: {result}")
            context_parts.append("")

        context_parts.append(
            "Adjust the plan for remaining work. Do NOT repeat steps that already succeeded."
        )

    system_content = _PLANNER_SYSTEM
    if context_parts:
        system_content += "\n" + "\n".join(context_parts)

    # Prepend skill instructions when a skill was loaded from metadata.
    skill_instructions = state.get("skill_instructions", "")
    if skill_instructions:
        system_content = skill_instructions + "\n\n" + system_content

    plan_messages = [SystemMessage(content=system_content)] + messages
    response = await llm.ainvoke(plan_messages)

    usage = getattr(response, 'usage_metadata', None) or {}
    prompt_tokens = usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)

    plan = _parse_plan(response.content)
    plan_version = state.get("plan_version", 0) + 1
    new_plan_steps = _make_plan_steps(plan, iteration=iteration)

    logger.info("Planner produced %d steps (iteration %d, version %d): %s",
                len(plan), iteration, plan_version, plan)

    return {
        "messages": [response],
        "plan": plan,
        "plan_steps": new_plan_steps,
        "plan_version": plan_version,
        "current_step": 0,
        "iteration": iteration + 1,
        "done": False,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "_system_prompt": system_content[:3000],
        "_prompt_messages": _summarize_messages(plan_messages),
    }


MAX_TOOL_CALLS_PER_STEP = int(_os.environ.get("SANDBOX_MAX_TOOL_CALLS_PER_STEP", "20"))


async def executor_node(
    state: dict[str, Any],
    llm_with_tools: Any,
) -> dict[str, Any]:
    """Execute the current plan step using the LLM with bound tools."""
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    tool_call_count = state.get("_tool_call_count", 0)

    if current_step >= len(plan):
        # No more steps — signal completion to reflector
        return {
            "messages": [AIMessage(content="All plan steps completed.")],
            "done": True,
        }

    # Guard: too many tool calls for this step — force completion
    if tool_call_count >= MAX_TOOL_CALLS_PER_STEP:
        logger.warning(
            "Step %d hit tool call limit (%d/%d) — forcing step completion",
            current_step, tool_call_count, MAX_TOOL_CALLS_PER_STEP,
        )
        return {
            "messages": [AIMessage(content=f"Step {current_step + 1} reached tool call limit ({MAX_TOOL_CALLS_PER_STEP}). Moving to reflection.")],
            "_tool_call_count": 0,
        }

    step_text = plan[current_step]
    system_content = _safe_format(
        _EXECUTOR_SYSTEM,
        current_step=current_step + 1,
        step_text=step_text,
        tool_call_count=tool_call_count,
        max_tool_calls=MAX_TOOL_CALLS_PER_STEP,
    )

    # Prepend skill instructions when a skill was loaded from metadata.
    skill_instructions = state.get("skill_instructions", "")
    if skill_instructions:
        system_content = skill_instructions + "\n\n" + system_content

    # Include the conversation history so the executor has full context
    messages = [SystemMessage(content=system_content)] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)

    # Track no-tool executions — if the LLM produces text instead of
    # tool calls, increment counter. After 2 consecutive no-tool runs
    # for the same step, mark the step as failed and advance.
    no_tool_count = state.get("_no_tool_count", 0)

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

    # -- Enforce single tool call (micro-reflection pattern) -------------------
    # Keep only the first tool call so the LLM sees each result before
    # deciding the next action. This prevents blind batching of N commands.
    if len(response.tool_calls) > 1:
        logger.info(
            "Executor returned %d tool calls — keeping only the first (micro-reflection)",
            len(response.tool_calls),
        )
        response = AIMessage(
            content=response.content,
            tool_calls=[response.tool_calls[0]],
        )

    # -- Detect unparsed text tool call attempts (stall signal) ----------------
    # If the model wrote text that looks like a tool call but wasn't parsed,
    # log a warning. The reflector will catch the zero-tool-call pattern.
    if not response.tool_calls and pre_patch_content:
        text_hint = str(pre_patch_content).lower()
        if any(kw in text_hint for kw in ("shell(", "file_read(", "file_write(",
                                            "```bash", "```shell", "i would run",
                                            "i will execute", "let me run")):
            logger.warning(
                "Executor produced text resembling a tool call but no actual "
                "tool_calls were generated — likely a stalled iteration"
            )

    # -- Dedup: skip tool calls that already have ToolMessage responses ------
    # The text-based parser generates fresh UUIDs each invocation, so
    # LangGraph treats re-parsed calls as new work.  Match on (name, args)
    # against already-executed calls in the CURRENT plan iteration to break
    # the executor→tools→executor loop.
    #
    # IMPORTANT: Only dedup within the current iteration (since the last
    # planner/replanner message). After a replan, the executor must be free
    # to retry the same tools — the new plan may need the same commands
    # to succeed with different context.
    if response.tool_calls:
        executed: set[tuple[str, str]] = set()
        messages = state.get("messages", [])

        # Find the boundary: start scanning from the last planner output.
        # Messages before that are from previous plan iterations and should
        # NOT cause dedup — the new plan may legitimately retry them.
        scan_start = 0
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            content = getattr(msg, "content", "")
            if isinstance(content, str) and "Plan:" in content and "Step " in content:
                scan_start = i
                break

        # Build a map from tool_call_id → (name, args) for AIMessage
        # tool calls SINCE the last planner output.
        tc_id_to_key: dict[str, tuple[str, str]] = {}
        for msg in messages[scan_start:]:
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
                # All calls already executed — signal reflector to advance
                # or replan rather than looping back to tools.
                logger.info(
                    "All tool calls deduped for step %d — signaling step complete",
                    state.get("current_step", 0),
                )
                return {
                    "messages": [
                        AIMessage(content=_DEDUP_SENTINEL)
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

    # If no tool calls after patching, the executor is either:
    # (a) Legitimately done with the step (summarizing results) — NORMAL
    # (b) Stalled and unable to call tools — only if it never called ANY tool
    #
    # With micro-reflection, the executor may produce text after a failed
    # tool call to summarize/report — that's valid step completion, not a stall.
    if not response.tool_calls:
        if tool_call_count > 0:
            # Executor already called tools this step — text response means
            # it's done summarizing. This is normal completion, not a stall.
            logger.info(
                "Executor produced text response after %d tool calls for step %d — step complete",
                tool_call_count, current_step,
            )
        else:
            no_tool_count += 1
            logger.warning(
                "Executor produced no tool calls for step %d (attempt %d/2)",
                current_step, no_tool_count,
            )
            if no_tool_count >= 2:
                logger.warning("Executor failed to call tools after 2 attempts — marking step failed")
                return {
                    "messages": [AIMessage(content=f"Step {current_step + 1} failed: executor could not call tools after 2 attempts.")],
                    "done": True if current_step + 1 >= len(plan) else False,
                    "_no_tool_count": 0,
                }
    else:
        no_tool_count = 0  # reset on successful tool call

    # Increment tool call count for micro-reflection tracking
    new_tool_call_count = tool_call_count + len(response.tool_calls)

    result: dict[str, Any] = {
        "messages": [response],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "_system_prompt": system_content[:3000],
        "_prompt_messages": _summarize_messages(messages),
        "_no_tool_count": no_tool_count,
        "_tool_call_count": new_tool_call_count,
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
    replan_count = state.get("replan_count", 0)
    done = state.get("done", False)
    recent_decisions = list(state.get("recent_decisions", []))

    # If executor signaled done (ran out of steps), go straight to done
    if done:
        return {"done": True}

    def _force_done(reason: str) -> dict[str, Any]:
        """Helper for early termination — marks current step failed, rest skipped."""
        ps = list(state.get("plan_steps", []))
        if current_step < len(ps):
            ps[current_step] = {**ps[current_step], "status": "failed"}
        for i in range(current_step + 1, len(ps)):
            if ps[i].get("status") == "pending":
                ps[i] = {**ps[i], "status": "skipped"}
        logger.warning("%s — forcing done", reason)
        return {
            "step_results": step_results,
            "plan_steps": ps,
            "current_step": current_step + 1,
            "done": True,
            "replan_count": replan_count,
        }

    # Budget guard — force termination if iterations exceeded
    if iteration >= budget.max_iterations:
        return _force_done(f"Budget exceeded: {iteration}/{budget.max_iterations} iterations used")

    # Count tool calls in this iteration (from executor's last message)
    messages = state["messages"]
    tool_calls_this_iter = 0
    last_content = ""
    if messages:
        last_msg = messages[-1]
        tool_calls_this_iter = len(getattr(last_msg, "tool_calls", []) or [])
        content = getattr(last_msg, "content", "")
        if isinstance(content, list):
            last_content = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            last_content = str(content)

    # Stall detection — force done if agent is stuck
    # Only count decisions AFTER the most recent replan (replans reset context)
    decisions_since_replan = []
    for d in reversed(recent_decisions):
        if d == "replan":
            break
        decisions_since_replan.insert(0, d)

    # 1. Two consecutive no-tool iterations since last replan → stuck
    no_tool_recent = 0
    for d in reversed(decisions_since_replan[-3:]):
        if d in ("replan", "continue"):
            no_tool_recent += 1
        else:
            break
    if no_tool_recent >= 2 and tool_calls_this_iter == 0:
        return _force_done(f"Stall: {no_tool_recent + 1} consecutive iterations with 0 tool calls")

    # 2. Identical executor output across 2 consecutive iterations → stuck
    if step_results and last_content[:500] == step_results[-1]:
        return _force_done("Stall: executor output identical to previous iteration")

    # If last_content is the dedup sentinel, recover the actual last tool
    # result from the message history so the reflector sees real output.
    if _DEDUP_SENTINEL in last_content:
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                last_content = str(getattr(msg, "content", ""))
                logger.info("Reflector: substituted dedup sentinel with last tool result (%d chars)",
                            len(last_content))
                break

    step_results.append(last_content[:500])

    step_text = plan[current_step] if current_step < len(plan) else "N/A"
    plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
    results_text = last_content[:1000]

    # Hint: if the step result contains error signals, prepend a note
    error_signals = ("error", "fatal", "failed", "exit_code", "stderr", "denied", "cannot")
    if any(sig in results_text.lower() for sig in error_signals):
        results_text = (
            "[NOTE: The step result below contains error indicators. "
            "Consider 'replan' to try a different approach.]\n\n" + results_text
        )

    # Build replan history context — show the LLM what prior replans tried
    replan_history_text = ""
    if replan_count > 0:
        replan_history_lines = [
            f"REPLAN HISTORY ({replan_count} prior replan(s)):"
        ]
        # Collect failed step summaries from plan_steps
        for ps in state.get("plan_steps", []):
            if ps.get("status") == "failed":
                summary = ps.get("result_summary", "no details")
                replan_history_lines.append(
                    f"  - Step {ps.get('index', '?')+1} FAILED: {ps.get('description', '?')[:80]}"
                    f" — {summary[:150]}"
                )
        replan_history_lines.append(
            "Do NOT repeat approaches that already failed. Try something fundamentally different,"
            " or choose 'done' to report partial results."
        )
        replan_history_text = "\n".join(replan_history_lines)

    # Ask LLM to reflect
    recent_str = ", ".join(recent_decisions[-5:]) if recent_decisions else "none"
    system_content = _safe_format(
        _REFLECTOR_SYSTEM,
        plan_text=plan_text,
        current_step=current_step + 1,
        step_text=step_text,
        step_result=results_text,
        iteration=iteration,
        max_iterations=budget.max_iterations,
        replan_count=replan_count,
        tool_calls_this_iter=tool_calls_this_iter,
        recent_decisions=recent_str,
        replan_history=replan_history_text,
    )
    reflect_messages = [SystemMessage(content=system_content)]
    response = await llm.ainvoke(reflect_messages)

    # Extract token usage from the LLM response
    usage = getattr(response, 'usage_metadata', None) or {}
    prompt_tokens = usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)

    decision = _parse_decision(response.content)
    recent_decisions.append(decision)
    recent_decisions = recent_decisions[-10:]

    # Update plan_steps with per-step status
    plan_steps = list(state.get("plan_steps", []))
    # Extract tool names used in this step from messages
    step_tools: list[str] = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", []) or []:
            name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
            if name not in step_tools:
                step_tools.append(name)

    if current_step < len(plan_steps):
        ps = {**plan_steps[current_step]}
        ps["tool_calls"] = step_tools
        ps["result_summary"] = last_content[:200]
        plan_steps[current_step] = ps

    logger.info(
        "Reflector decision: %s (step %d/%d, iter %d, replans=%d, tools=%d, recent=%s)",
        decision, current_step + 1, len(plan), iteration,
        replan_count, tool_calls_this_iter,
        recent_decisions[-3:],
    )

    base_result = {
        "messages": [response],
        "step_results": step_results,
        "recent_decisions": recent_decisions,
        "plan_steps": plan_steps,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "_system_prompt": system_content[:3000],
        "_prompt_messages": _summarize_messages(reflect_messages),
    }

    if decision == "done":
        # Mark current step done, remaining as skipped
        if current_step < len(plan_steps):
            plan_steps[current_step] = {**plan_steps[current_step], "status": "done"}
        for i in range(current_step + 1, len(plan_steps)):
            if plan_steps[i].get("status") == "pending":
                plan_steps[i] = {**plan_steps[i], "status": "skipped"}
        return {
            **base_result,
            "plan_steps": plan_steps,
            "current_step": current_step + 1,
            "done": True,
            "replan_count": replan_count,
        }
    elif decision == "replan":
        new_replan_count = replan_count + 1
        # Mark current step failed
        if current_step < len(plan_steps):
            plan_steps[current_step] = {**plan_steps[current_step], "status": "failed"}
        logger.info("Replan %d — routing back to planner", new_replan_count)
        return {
            **base_result,
            "plan_steps": plan_steps,
            "done": False,
            "replan_count": new_replan_count,
        }
    else:
        # Continue: mark current step done, advance
        if current_step < len(plan_steps):
            plan_steps[current_step] = {**plan_steps[current_step], "status": "done"}
        next_step = current_step + 1
        if next_step < len(plan_steps):
            plan_steps[next_step] = {**plan_steps[next_step], "status": "running"}
        if next_step >= len(plan):
            logger.info(
                "All %d planned steps completed — routing to planner for reassessment",
                len(plan),
            )
            return {
                **base_result,
                "plan_steps": plan_steps,
                "done": False,
                "replan_count": replan_count,
                "_tool_call_count": 0,
            }
        return {
            **base_result,
            "plan_steps": plan_steps,
            "current_step": next_step,
            "done": False,
            "replan_count": replan_count,
            "_tool_call_count": 0,
        }


async def reporter_node(
    state: dict[str, Any],
    llm: Any,
) -> dict[str, Any]:
    """Format accumulated step results into a final answer.

    Sets ``plan_status`` based on how the loop ended:
    - All steps done → ``"completed"``
    - Stall/budget forced done → ``"failed"`` (with ``awaiting_continue``
      so user/looper can retry)
    - Plan steps remain → ``"awaiting_continue"``
    """
    plan = state.get("plan", [])
    step_results = state.get("step_results", [])
    plan_steps = state.get("plan_steps", [])

    # Determine terminal plan_status based on step outcomes
    if plan_steps:
        done_count = sum(1 for s in plan_steps if s.get("status") == "done")
        failed_count = sum(1 for s in plan_steps if s.get("status") == "failed")
        total = len(plan_steps)
        if done_count == total:
            terminal_status = "completed"
        elif failed_count > 0 or done_count < total:
            terminal_status = "awaiting_continue"
        else:
            terminal_status = "completed"
    else:
        terminal_status = "completed"

    # Filter out internal dedup sentinel from step_results so it never
    # reaches the reporter prompt or the final answer.
    step_results = [r for r in step_results if _DEDUP_SENTINEL not in r]

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
            # Guard: skip internal dedup sentinel — fall through to
            # LLM-based summary which uses real step_results instead.
            if _DEDUP_SENTINEL in text:
                pass  # fall through
            # Guard: if text is a bare reflector decision keyword
            # (e.g. budget exhaustion forces done with "continue"),
            # fall through to LLM-based summary from step_results.
            elif not _BARE_DECISION_RE.match(text.strip()):
                return {"final_answer": text, "plan_status": terminal_status}
            # Fall through to LLM-based summary below
        elif not step_results:
            return {"final_answer": "No response generated.", "plan_status": terminal_status}

    plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
    results_text = "\n".join(
        f"Step {i+1}: {r}" for i, r in enumerate(step_results)
    )

    # Build step status summary from plan_steps
    step_status_lines = []
    for ps in plan_steps:
        idx = ps.get("index", 0)
        status = ps.get("status", "unknown").upper()
        desc = ps.get("description", "")[:80]
        result = ps.get("result_summary", "")[:100]
        line = f"{idx+1}. [{status}] {desc}"
        if result and status == "failed":
            line += f" — ERROR: {result}"
        step_status_lines.append(line)
    step_status_text = "\n".join(step_status_lines) if step_status_lines else "No step status available."

    system_content = _safe_format(
        _REPORTER_SYSTEM,
        plan_text=plan_text,
        step_status_text=step_status_text,
        results_text=results_text,
    )
    # Filter dedup sentinel messages from conversation history passed to the
    # reporter LLM so it cannot echo them in the final answer.
    filtered_msgs = [
        m for m in state["messages"]
        if _DEDUP_SENTINEL not in str(getattr(m, "content", ""))
    ]
    messages = [SystemMessage(content=system_content)] + filtered_msgs
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

    logger.info("Reporter: plan_status=%s (done=%d, failed=%d, total=%d)",
                terminal_status,
                sum(1 for s in plan_steps if s.get("status") == "done"),
                sum(1 for s in plan_steps if s.get("status") == "failed"),
                len(plan_steps))

    return {
        "messages": [response],
        "final_answer": text,
        "plan_status": terminal_status,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "_system_prompt": system_content[:3000],
        "_prompt_messages": _summarize_messages(messages),
    }


# ---------------------------------------------------------------------------
# Routing function for reflector conditional edges
# ---------------------------------------------------------------------------


def route_reflector(state: dict[str, Any]) -> str:
    """Route from reflector based on decision.

    ``done``     → reporter (final answer)
    ``replan``   → planner (create new plan)
    ``continue`` → executor (execute next step)
    """
    if state.get("done", False):
        return "done"
    # Check the reflector's decision to distinguish continue vs replan
    decision = (state.get("recent_decisions") or ["continue"])[-1]
    if decision == "replan":
        return "replan"
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
