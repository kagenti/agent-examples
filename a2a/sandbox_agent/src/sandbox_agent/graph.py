"""LangGraph agent graph with plan-execute-reflect reasoning loop.

The graph binds six tools to an LLM and uses a structured reasoning loop:

- **shell**: runs commands via :class:`SandboxExecutor` (with permission checks)
- **file_read**: reads files relative to the workspace (prevents path traversal)
- **file_write**: writes files relative to the workspace (prevents path traversal)
- **web_fetch**: fetches web content from allowed domains
- **explore**: spawns a read-only sub-agent for codebase research
- **delegate**: spawns a child agent session for delegated tasks

Graph architecture (router → plan → execute → reflect):

```mermaid
graph TD
    START((User Message)) --> router
    router -->|new/replan| planner
    router -->|resume| executor

    planner --> executor
    executor -->|tool_calls| tools
    tools --> executor
    executor -->|no tool_calls| reflector

    reflector -->|execute| executor
    reflector -->|replan| planner
    reflector -->|done| reporter
    reporter --> END((Final Answer))

    style router fill:#4CAF50,color:white
    style planner fill:#2196F3,color:white
    style executor fill:#FF9800,color:white
    style tools fill:#607D8B,color:white
    style reflector fill:#9C27B0,color:white
    style reporter fill:#F44336,color:white
```

Key flows:
- **execute**: Step succeeded → executor runs the next plan step
- **replan**: Step failed → planner creates a new plan → executor runs it
- **done**: Task complete → reporter summarizes results

The executor uses micro-reflection: one tool call per LLM invocation,
see result, decide next action. Budget limits (iterations, tokens,
wall clock) are the only hard stops.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Send, interrupt

try:
    from langgraph.errors import GraphInterrupt
except ImportError:
    # Fallback for older langgraph versions
    GraphInterrupt = type("GraphInterrupt", (Exception,), {})

from sandbox_agent.budget import AgentBudget
from sandbox_agent.executor import HitlRequired, SandboxExecutor
from sandbox_agent.permissions import PermissionChecker
from sandbox_agent.reasoning import (
    PlanStep,
    _DEBUG_PROMPTS,
    executor_node,
    planner_node,
    reflector_node,
    reporter_node,
    route_entry,
    route_reflector,
    router_node,
)
from sandbox_agent.sources import SourcesConfig
from sandbox_agent.subagents import make_delegate_tool, make_explore_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class SandboxState(MessagesState):
    """Extended MessagesState carrying sandbox-specific fields.

    Attributes
    ----------
    context_id:
        A2A context identifier for multi-turn conversations.
    workspace_path:
        Absolute path to the per-context workspace directory.
    final_answer:
        The agent's final answer (set when the graph completes).
    plan:
        Flat list of step descriptions (backward compat with serializer).
    plan_steps:
        Structured per-step tracking with status, tool calls, results.
        This is the source of truth; ``plan`` is derived from it.
    plan_status:
        Lifecycle status of the plan across A2A turns:
        ``"executing"`` | ``"completed"`` | ``"failed"`` | ``"awaiting_continue"``
    plan_version:
        Incremented on each replan.
    original_request:
        The user's first message that created this plan.
    current_step:
        Index of the plan step currently being executed (0-based).
    step_results:
        Summary of each completed step's output.
    iteration:
        Outer-loop iteration counter (planner → executor → reflector).
    replan_count:
        Number of times the reflector has chosen "replan". Used to cap
        the replan loop and force termination after MAX_REPLAN_COUNT.
    done:
        Flag set by reflector when the task is complete.
    skill_instructions:
        Optional skill content loaded from a ``.claude/skills/`` file.
    recent_decisions:
        Rolling window of the last 10 reflector decisions (continue/replan/done).
    _route:
        Internal routing signal from the router node (not persisted).
    """

    context_id: str
    workspace_path: str
    final_answer: str
    plan: list[str]
    plan_steps: list[PlanStep]
    plan_status: str
    plan_version: int
    original_request: str
    current_step: int
    step_results: list[str]
    iteration: int
    replan_count: int
    done: bool
    skill_instructions: str
    prompt_tokens: int
    completion_tokens: int
    recent_decisions: list[str]
    _tool_call_count: int
    _route: str
    _system_prompt: str
    _prompt_messages: list[dict]
    _budget_summary: dict
    _no_tool_count: int
    model: str


# ---------------------------------------------------------------------------
# Skill loader
# ---------------------------------------------------------------------------


def _load_skill(workspace: str, skill_id: str) -> str | None:
    """Load a skill file from the workspace's ``.claude/skills/`` directory.

    Parameters
    ----------
    workspace:
        Absolute path to the workspace root (or repo root).
    skill_id:
        Skill identifier, e.g. ``"rca:ci"`` or ``"tdd:hypershift"``.
        Colons are converted to directory separators so ``rca:ci``
        resolves to ``rca/ci.md``.

    Returns
    -------
    str | None
        The skill file content, or ``None`` if no matching file exists.
    """
    # Search in multiple locations:
    # 1. Per-session workspace: /workspace/{contextId}/.claude/skills/
    # 2. Shared workspace root: /workspace/.claude/skills/ (cloned at startup)
    workspace_root = os.environ.get("WORKSPACE_DIR", "/workspace")
    search_dirs = [
        Path(workspace) / ".claude" / "skills",
        Path(workspace_root) / ".claude" / "skills",
    ]

    for skills_dir in search_dirs:
        if not skills_dir.is_dir():
            continue

        # Primary path: replace ':' with '/' → rca:ci → rca/ci.md
        primary = skills_dir / f"{skill_id.replace(':', '/')}.md"
        if primary.is_file():
            logger.info("Loaded skill '%s' from %s", skill_id, primary)
            return primary.read_text(encoding="utf-8", errors="replace")

        # Try SKILL.md inside directory named with colons → rca:ci/SKILL.md
        skill_dir = skills_dir / skill_id.replace(":", "/")
        skill_md = skill_dir / "SKILL.md"
        if skill_md.is_file():
            logger.info("Loaded skill '%s' from %s", skill_id, skill_md)
            return skill_md.read_text(encoding="utf-8", errors="replace")

        # Directory named with literal colon → rca:ci/SKILL.md
        colon_dir = skills_dir / skill_id
        colon_skill = colon_dir / "SKILL.md"
        if colon_skill.is_file():
            logger.info("Loaded skill '%s' from %s (colon dir)", skill_id, colon_skill)
            return colon_skill.read_text(encoding="utf-8", errors="replace")

    logger.warning("Skill '%s' not found in any search path", skill_id)
    return None


# ---------------------------------------------------------------------------
# Tool factories
# ---------------------------------------------------------------------------


def _make_shell_tool(executor: SandboxExecutor) -> Any:
    """Return a LangChain tool that delegates to *executor.run_shell*.

    On :class:`HitlRequired`, the tool calls LangGraph ``interrupt()`` to
    pause the graph and require explicit human approval before resuming.
    The graph will not continue until the human responds.
    """

    @tool
    async def shell(command: str) -> str:
        """Execute a shell command in the session workspace.

        The working directory is the per-session workspace. Use relative
        paths for files in this session. Files created here are visible
        in the Files tab.

        Args:
            command: The shell command to run.

        Returns:
            Command output (stdout + stderr), or pauses for human approval.
        """
        # Warn on bare `cd` — it has no effect in isolated shell execution
        if command.strip().startswith("cd ") and "&&" not in command:
            logger.warning(
                "Bare 'cd' command detected — has no effect in isolated shell: %s",
                command,
            )

        try:
            result = await executor.run_shell(command)
        except HitlRequired as exc:
            # Pause graph execution — requires human approval to resume.
            # The interrupt() call suspends the graph state. The A2A task
            # transitions to input_required. Only an explicit human
            # approval (via the HITLManager channel) resumes execution.
            approval = interrupt({
                "type": "approval_required",
                "command": exc.command,
                "message": f"Command '{exc.command}' requires human approval.",
            })
            # If we reach here, the human approved — execute the command.
            if isinstance(approval, dict) and approval.get("approved"):
                result = await executor._execute(command)
            else:
                return f"DENIED: command '{exc.command}' was rejected by human review."

        # Retry on rate-limit errors (GitHub API, etc.) with exponential backoff
        output = _format_result(result)
        if result.exit_code != 0 and _is_rate_limited(output):
            import asyncio
            for attempt in range(1, 4):  # up to 3 retries
                delay = 2 ** attempt  # 2s, 4s, 8s
                logger.info("Rate limit detected, retry %d/3 after %ds", attempt, delay)
                await asyncio.sleep(delay)
                try:
                    result = await executor.run_shell(command)
                except HitlRequired:
                    break  # don't retry HITL
                output = _format_result(result)
                if result.exit_code == 0 or not _is_rate_limited(output):
                    break

        return output

    return shell


_MAX_TOOL_OUTPUT = 10_000  # chars — prevent context window blowout


def _format_result(result: Any) -> str:
    """Format an ExecutionResult into a string, truncating large output."""
    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        if result.exit_code != 0:
            parts.append(f"STDERR: {result.stderr}")
        else:
            # Informational stderr (e.g., git clone progress) — not an error
            parts.append(result.stderr)
    if result.exit_code != 0:
        parts.append(f"EXIT_CODE: {result.exit_code}")
    text = "\n".join(parts) if parts else "(no output)"
    if len(text) > _MAX_TOOL_OUTPUT:
        kept = text[:_MAX_TOOL_OUTPUT]
        dropped = len(text) - _MAX_TOOL_OUTPUT
        text = f"{kept}\n\n[OUTPUT TRUNCATED — {dropped:,} chars omitted. Redirect large output to a file: command > output/result.txt]"
    return text


def _is_rate_limited(output: str) -> bool:
    """Detect rate-limit errors in command output."""
    lower = output.lower()
    return any(pattern in lower for pattern in (
        "rate limit exceeded",
        "rate limit",
        "too many requests",
        "429",
        "api rate limit",
        "secondary rate limit",
    ))


def _make_file_read_tool(workspace_path: str) -> Any:
    """Return a LangChain tool that reads files relative to *workspace_path*.

    The tool prevents path traversal by resolving the path and ensuring it
    stays within the workspace directory.
    """
    ws_root = Path(workspace_path).resolve()

    @tool
    async def file_read(path: str) -> str:
        """Read a file from the workspace.

        Args:
            path: Relative path within the workspace directory.

        Returns:
            The file contents, or an error message.
        """
        resolved = (ws_root / path).resolve()

        # Prevent path traversal.
        if not resolved.is_relative_to(ws_root):
            return f"Error: path '{path}' resolves outside the workspace."

        if not resolved.is_file():
            return f"Error: file not found at '{path}'."

        try:
            return resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"Error reading file: {exc}"

    return file_read


def _make_file_write_tool(workspace_path: str) -> Any:
    """Return a LangChain tool that writes files relative to *workspace_path*.

    The tool prevents path traversal and creates parent directories as needed.
    """
    ws_root = Path(workspace_path).resolve()

    @tool
    async def file_write(path: str, content: str) -> str:
        """Write content to a file in the workspace.

        Args:
            path: Relative path within the workspace directory.
            content: The text content to write.

        Returns:
            A confirmation message, or an error message.
        """
        resolved = (ws_root / path).resolve()

        # Prevent path traversal.
        if not resolved.is_relative_to(ws_root):
            return f"Error: path '{path}' resolves outside the workspace."

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to '{path}'."
        except OSError as exc:
            return f"Error writing file: {exc}"

    return file_write


def _make_grep_tool(workspace_path: str) -> Any:
    """Return a LangChain tool that searches file contents with regex."""
    ws_root = Path(workspace_path).resolve()

    @tool
    async def grep(pattern: str, path: str = ".", include: str = "") -> str:
        """Search for a regex pattern in file contents under the workspace.

        Args:
            pattern: Regex pattern to search for (e.g. 'def main', 'ERROR|FAIL').
            path: Relative directory or file to search in (default: workspace root).
            include: Glob filter for filenames (e.g. '*.py', '*.ts'). Empty = all files.

        Returns:
            Matching lines with file paths and line numbers, or an error message.
        """
        import asyncio as _aio

        search_path = (ws_root / path).resolve()
        if not search_path.is_relative_to(ws_root):
            return f"Error: path '{path}' resolves outside the workspace."

        cmd = ["grep", "-rn", "--color=never"]
        if include:
            cmd.extend(["--include", include])
        cmd.extend([pattern, str(search_path)])

        try:
            proc = await _aio.create_subprocess_exec(
                *cmd, stdout=_aio.subprocess.PIPE, stderr=_aio.subprocess.PIPE,
            )
            stdout, stderr = await _aio.wait_for(proc.communicate(), timeout=30)
            out = stdout.decode(errors="replace")[:10000]
            if proc.returncode == 1:
                return "No matches found."
            if proc.returncode != 0:
                return f"Error: {stderr.decode(errors='replace')[:500]}"
            # Make paths relative to workspace
            return out.replace(str(ws_root) + "/", "")
        except Exception as exc:
            return f"Error running grep: {exc}"

    return grep


def _make_glob_tool(workspace_path: str) -> Any:
    """Return a LangChain tool that finds files by glob pattern."""
    ws_root = Path(workspace_path).resolve()

    @tool
    async def glob(pattern: str) -> str:
        """Find files matching a glob pattern in the workspace.

        Args:
            pattern: Glob pattern (e.g. '**/*.py', 'src/**/*.ts', '*.md').

        Returns:
            Newline-separated list of matching file paths relative to workspace.
        """
        import fnmatch
        matches = []
        for p in sorted(ws_root.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(ws_root))
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(p.name, pattern):
                    matches.append(rel)
                    if len(matches) >= 200:
                        matches.append(f"... truncated ({len(matches)}+ matches)")
                        break
        return "\n".join(matches) if matches else "No files matched."

    return glob


def _make_web_fetch_tool(sources_config: SourcesConfig) -> Any:
    """Return a LangChain tool that fetches web content from allowed domains.

    The tool checks the URL's domain against ``sources.json`` allowed_domains
    before making the request.
    """

    @tool
    async def web_fetch(url: str) -> str:
        """Fetch content from a URL.

        Domain filtering is handled by the outbound Squid proxy at the
        network level. This tool fetches any URL the proxy allows.

        Args:
            url: The full URL to fetch (e.g. https://github.com/org/repo/issues/1).

        Returns:
            The page content as text, or an error message.
        """
        import httpx
        from urllib.parse import urlparse

        parsed = urlparse(url)
        domain = parsed.hostname or ""

        if not sources_config.is_web_access_enabled():
            return "Error: web access is disabled in sources.json."

        # Domain filtering is delegated to the Squid proxy.
        # Log the domain for observability but don't block.
        logger.info("web_fetch: domain=%s url=%s", domain, url[:200])

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "kagenti-sandbox-agent/1.0"})
                resp.raise_for_status()

                content_type = resp.headers.get("content-type", "")
                text = resp.text

                # For HTML, try to extract readable text
                if "text/html" in content_type:
                    # Simple HTML tag stripping for readability
                    import re
                    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
                    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()

                # Truncate very long responses
                if len(text) > 50000:
                    text = text[:50000] + "\n\n[Content truncated at 50000 characters]"

                return text

        except httpx.HTTPStatusError as exc:
            return f"Error: HTTP {exc.response.status_code} fetching {url}"
        except httpx.RequestError as exc:
            return f"Error: could not fetch {url}: {exc}"

    return web_fetch


# ---------------------------------------------------------------------------
# Escape tool for Llama 4 Scout
# ---------------------------------------------------------------------------
# Llama 4 Scout ALWAYS calls a tool when tools are bound (tool_choice=auto
# acts like required). The respond_to_user tool lets planner/reflector
# "escape" the tool loop by calling this tool with their final text output.


@tool
def respond_to_user(response: str) -> str:
    """Return your final text response. Call this when you have enough
    information and don't need any more tools.

    Args:
        response: The complete text response to return to the user.

    Returns:
        The response text unchanged.
    """
    return response


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(
    workspace_path: str,
    permission_checker: PermissionChecker,
    sources_config: SourcesConfig,
    checkpointer: Optional[Any] = None,
    context_id: str = "",
    namespace: str = "team1",
) -> Any:
    """Build and compile the LangGraph agent graph.

    Parameters
    ----------
    workspace_path:
        Absolute path to the per-context workspace directory.
    permission_checker:
        A :class:`PermissionChecker` for evaluating shell operations.
    sources_config:
        A :class:`SourcesConfig` providing runtime limits.
    checkpointer:
        Optional LangGraph checkpointer for PostgreSQL-based state
        persistence across A2A turns.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph with ``ainvoke`` / ``astream`` methods.
    """
    # -- Executor -----------------------------------------------------------
    executor = SandboxExecutor(
        workspace_path=workspace_path,
        permission_checker=permission_checker,
        sources_config=sources_config,
    )

    # -- LLM ----------------------------------------------------------------
    from sandbox_agent.configuration import Configuration

    config = Configuration()  # type: ignore[call-arg]
    # -- Budget -------------------------------------------------------------
    budget = AgentBudget()

    llm = ChatOpenAI(
        model=config.llm_model,
        base_url=config.llm_api_base,
        api_key=config.llm_api_key,
        timeout=budget.llm_timeout,
        max_retries=budget.llm_max_retries,
        model_kwargs={
            "extra_body": {
                "metadata": {
                    "session_id": context_id,
                    "agent_name": os.environ.get("AGENT_NAME", "sandbox-legion"),
                    "namespace": namespace,
                    "max_session_tokens": budget.max_tokens,
                }
            }
        },
    )

    # -- Tools --------------------------------------------------------------
    # Create tool instances once — shared across node subsets.
    shell_tool = _make_shell_tool(executor)
    file_read_tool = _make_file_read_tool(workspace_path)
    file_write_tool = _make_file_write_tool(workspace_path)
    grep_tool = _make_grep_tool(workspace_path)
    glob_tool = _make_glob_tool(workspace_path)
    web_fetch_tool = _make_web_fetch_tool(sources_config)

    core_tools = [shell_tool, file_read_tool, file_write_tool, grep_tool, glob_tool, web_fetch_tool]
    tools = core_tools + [
        make_explore_tool(workspace_path, llm),
        # delegate disabled — causes crashes when agent can't resolve paths
        # make_delegate_tool(workspace_path, llm, context_id, core_tools, namespace),
    ]

    # -- Per-node tool subsets ------------------------------------------------
    # Each reasoning node gets its own tools and tool_choice mode:
    #   executor:  ALL tools, tool_choice="any" (must call tools)
    #   planner:   glob, grep, file_read, file_write + respond_to_user (escape)
    #   reflector: glob, grep, file_read + respond_to_user (escape)
    #   router/reporter/step_selector: no tools (text-only)

    read_only_tools = [file_read_tool, grep_tool, glob_tool, respond_to_user]
    planner_tools = [file_read_tool, grep_tool, glob_tool, file_write_tool, respond_to_user]

    # Executor uses tool_choice="any" — MUST call tools (not produce text).
    # Planner and reflector use "auto" — CAN choose not to call tools.
    llm_executor = llm.bind_tools(tools, tool_choice="any")
    llm_planner = llm.bind_tools(planner_tools)  # defaults to auto

    # All nodes with tools use tool_choice="auto"
    llm_reflector = llm.bind_tools(read_only_tools)  # read-only for verification

    # ToolNodes for each node's tool subset
    _executor_tool_node = ToolNode(tools)
    _planner_tool_node = ToolNode(planner_tools)
    _reflector_tool_node = ToolNode(read_only_tools)

    # -- Graph nodes (router-plan-execute-reflect) ---------------------------
    # Each node function from reasoning.py takes (state, llm) — we wrap them
    # in closures that capture the appropriate LLM instance.

    async def _router(state: SandboxState) -> dict[str, Any]:
        return await router_node(state)

    async def _planner(state: SandboxState) -> dict[str, Any]:
        return await planner_node(state, llm_planner, budget=budget)

    async def _executor(state: SandboxState) -> dict[str, Any]:
        return await executor_node(state, llm_executor, budget=budget)

    async def _reflector(state: SandboxState) -> dict[str, Any]:
        return await reflector_node(state, llm_reflector, budget=budget)

    async def _reporter(state: SandboxState) -> dict[str, Any]:
        return await reporter_node(state, llm, budget=budget)

    async def _step_selector(state: SandboxState) -> dict[str, Any]:
        """Pick the next step and prepare focused context for the executor.

        Uses a lightweight LLM call to review plan progress and write
        a targeted brief for the executor — what to do, what worked/failed
        before, and what to avoid.
        """
        from langchain_core.messages import SystemMessage as SM, HumanMessage as HM

        plan = state.get("plan", [])
        plan_steps = list(state.get("plan_steps", []))
        current = state.get("current_step", 0)
        messages = state.get("messages", [])

        # Find next non-done step
        next_step = current
        for i in range(current, len(plan_steps)):
            ps = plan_steps[i]
            status = ps.get("status", "pending") if isinstance(ps, dict) else "pending"
            if status != "done":
                next_step = i
                break
        else:
            next_step = len(plan)

        # Mark selected step as running
        if next_step < len(plan_steps) and isinstance(plan_steps[next_step], dict):
            plan_steps[next_step] = {**plan_steps[next_step], "status": "running"}

        # Build plan status summary
        plan_summary = []
        for i, step in enumerate(plan):
            ps = plan_steps[i] if i < len(plan_steps) else {}
            status = ps.get("status", "pending") if isinstance(ps, dict) else "pending"
            marker = "✓" if status == "done" else "→" if i == next_step else " "
            result_hint = ""
            if isinstance(ps, dict) and ps.get("result_summary"):
                result_hint = f" — {ps['result_summary'][:100]}"
            plan_summary.append(f"  {marker} {i+1}. [{status}] {step[:80]}{result_hint}")

        # Gather recent tool results (last 3 ToolMessages)
        recent_results = []
        for m in reversed(messages[-10:]):
            if hasattr(m, 'name') and getattr(m, 'type', '') == 'tool':
                content = str(getattr(m, 'content', ''))[:300]
                recent_results.insert(0, f"  [{m.name}] {content}")
                if len(recent_results) >= 3:
                    break

        if next_step >= len(plan):
            # All done
            logger.info("StepSelector: all %d steps complete", len(plan))
            return {
                "current_step": next_step,
                "plan_steps": plan_steps,
                "_tool_call_count": 0,
                "done": True,
            }

        # Quick LLM call — write a focused brief for the executor
        step_text = plan[next_step] if next_step < len(plan) else "N/A"
        prompt = f"""You are a step coordinator. Write a 2-3 sentence brief for the executor.

Plan progress:
{chr(10).join(plan_summary)}

Next step to execute: {next_step + 1}. {step_text}

Recent tool results:
{chr(10).join(recent_results) if recent_results else '(none yet)'}

WORKSPACE RULE: Each shell command starts fresh in /workspace. Bare `cd` has no effect.
If the step involves a cloned repo, always write `cd repos/<repo> && <command>` in the brief.
Example: "cd repos/kagenti && gh pr list" — never just "gh pr list".

Write a brief: what EXACTLY to do for step {next_step + 1}, what context from previous steps is relevant, and what to watch out for. Be specific about commands/tools to use, and always include the full `cd <dir> && command` pattern when a cloned repo is involved."""

        sys_msg = SM(content="You are a concise step coordinator. Output ONLY the brief, no preamble.")
        user_msg = HM(content=prompt)
        try:
            response = await llm.ainvoke([sys_msg, user_msg])
            brief = response.content.strip()
            usage = getattr(response, 'usage_metadata', None) or {}
            budget.add_tokens(
                usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
            )
        except Exception as e:
            logger.warning("StepSelector LLM call failed: %s — using default brief", e)
            brief = f"Execute step {next_step + 1}: {step_text}"
            response = None

        logger.info("StepSelector: step %d/%d brief: %s", next_step + 1, len(plan), brief[:100])
        result: dict[str, Any] = {
            "current_step": next_step,
            "plan_steps": plan_steps,
            "_tool_call_count": 0,
            "skill_instructions": f"STEP BRIEF FROM COORDINATOR:\n{brief}\n\n---\n",
        }
        if _DEBUG_PROMPTS:
            from sandbox_agent.reasoning import _format_llm_response
            result["_system_prompt"] = prompt[:10000]
            if response:
                result["_llm_response"] = _format_llm_response(response)
        return result

    # -- Safe ToolNode wrappers — never crash the graph ----------------------

    def _make_safe_tool_wrapper(tool_node: ToolNode, label: str):
        """Create a safe tool execution wrapper for a ToolNode."""
        async def _safe(state: SandboxState) -> dict[str, Any]:
            from langchain_core.messages import ToolMessage
            try:
                return await tool_node.ainvoke(state)
            except (GraphInterrupt, KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                logger.error("%s ToolNode error: %s", label, exc, exc_info=True)
                messages = state.get("messages", [])
                error_msgs = []
                if messages:
                    last = messages[-1]
                    for tc in getattr(last, "tool_calls", []):
                        tc_id = tc.get("id", "unknown") if isinstance(tc, dict) else getattr(tc, "id", "unknown")
                        tc_name = tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                        error_msgs.append(ToolMessage(
                            content=f"Tool error: {exc}",
                            tool_call_id=tc_id,
                            name=tc_name,
                        ))
                if not error_msgs:
                    error_msgs.append(ToolMessage(
                        content=f"Tool execution failed: {exc}",
                        tool_call_id="error",
                        name="unknown",
                    ))
                return {"messages": error_msgs}
        return _safe

    _safe_executor_tools = _make_safe_tool_wrapper(_executor_tool_node, "executor")
    _safe_planner_tools = _make_safe_tool_wrapper(_planner_tool_node, "planner")
    _safe_reflector_tools = _make_safe_tool_wrapper(_reflector_tool_node, "reflector")

    # -- Assemble graph -----------------------------------------------------
    #
    # Topology (all nodes use tool_choice="auto"):
    #
    #   router → [plan]   → planner ⇄ planner_tools → step_selector
    #            [resume] → step_selector
    #
    #   step_selector → executor ⇄ tools → reflector ⇄ reflector_tools
    #
    #   reflector_route → [done]     → reporter → END
    #                     [continue] → step_selector
    #                     [replan]   → planner
    #
    # Tool subsets:
    #   planner:  glob, grep, file_read, file_write (inspect workspace, save plans)
    #   executor: all tools (shell, files, grep, glob, web_fetch, explore, delegate)
    #   reflector: glob, grep, file_read (verify step outcomes before deciding)
    #
    graph = StateGraph(SandboxState)
    graph.add_node("router", _router)
    graph.add_node("planner", _planner)
    graph.add_node("planner_tools", _safe_planner_tools)
    graph.add_node("step_selector", _step_selector)
    graph.add_node("executor", _executor)
    graph.add_node("tools", _safe_executor_tools)
    graph.add_node("reflector", _reflector)
    graph.add_node("reflector_tools", _safe_reflector_tools)
    graph.add_node("reporter", _reporter)

    # Entry: router decides resume vs plan
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_entry,
        {"resume": "step_selector", "plan": "planner"},
    )

    # Planner → planner_tools (if tool_calls) or → step_selector (if no tool_calls)
    graph.add_conditional_edges(
        "planner",
        tools_condition,
        {"tools": "planner_tools", "__end__": "step_selector"},
    )
    graph.add_edge("planner_tools", "planner")

    graph.add_edge("step_selector", "executor")

    # Executor → executor_tools (if tool_calls) or → reflector (if no tool_calls)
    graph.add_conditional_edges(
        "executor",
        tools_condition,
        {"tools": "tools", "__end__": "reflector"},
    )
    graph.add_edge("tools", "executor")

    # Reflector → reflector_tools (if tool_calls) or → route decision
    graph.add_conditional_edges(
        "reflector",
        tools_condition,
        {"tools": "reflector_tools", "__end__": "reflector_route"},
    )
    graph.add_edge("reflector_tools", "reflector")

    # Reflector route → reporter (done), step_selector (continue), or planner (replan)
    graph.add_node("reflector_route", lambda state: state)  # pass-through
    graph.add_conditional_edges(
        "reflector_route",
        route_reflector,
        {"done": "reporter", "execute": "step_selector", "replan": "planner"},
    )
    graph.add_edge("reporter", "__end__")

    return graph.compile(checkpointer=checkpointer)
