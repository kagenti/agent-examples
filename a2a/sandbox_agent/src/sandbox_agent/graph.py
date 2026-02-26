"""LangGraph agent graph with sandboxed shell, file_read, and file_write tools.

The graph binds three tools to an LLM:

- **shell**: runs commands via :class:`SandboxExecutor` (with permission checks)
- **file_read**: reads files relative to the workspace (prevents path traversal)
- **file_write**: writes files relative to the workspace (prevents path traversal)

The graph follows the standard LangGraph react-agent pattern:

    assistant  -->  tools  -->  assistant  -->  END
                  (conditional)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

from sandbox_agent.executor import HitlRequired, SandboxExecutor
from sandbox_agent.permissions import PermissionChecker
from sandbox_agent.sources import SourcesConfig
from sandbox_agent.subagents import make_delegate_tool, make_explore_tool

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
    """

    context_id: str
    workspace_path: str
    final_answer: str


# ---------------------------------------------------------------------------
# Tool factories
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a sandboxed coding assistant.  You can execute shell commands, \
read files, and write files inside the user's workspace directory.

Available tools:
- **shell**: Execute a shell command.  Some commands may be denied by policy \
or require human approval (HITL).
- **file_read**: Read a file from the workspace.  Provide a path relative to \
the workspace root.
- **file_write**: Write content to a file in the workspace.  Provide a \
relative path and the content.  Parent directories are created automatically.
- **web_fetch**: Fetch content from a URL.  Only allowed domains (configured \
in sources.json) can be accessed.  Use this to read GitHub issues, PRs, \
documentation, and other web resources.
- **explore**: Spawn a read-only sub-agent for codebase research. The \
sub-agent can grep, read files, and list files but cannot write or execute \
commands.  Use this for searching definitions, analyzing code, or gathering \
information across multiple files.
- **delegate**: Spawn a separate sandbox pod for isolated, long-running, or \
untrusted tasks.  Requires a Kubernetes cluster with agent-sandbox CRDs.

Always prefer using the provided tools rather than raw shell I/O for file \
operations when possible, as they have built-in path-safety checks.
"""


def _make_shell_tool(executor: SandboxExecutor) -> Any:
    """Return a LangChain tool that delegates to *executor.run_shell*.

    On :class:`HitlRequired`, the tool calls LangGraph ``interrupt()`` to
    pause the graph and require explicit human approval before resuming.
    The graph will not continue until the human responds.
    """

    @tool
    async def shell(command: str) -> str:
        """Execute a shell command in the sandbox workspace.

        Args:
            command: The shell command to run.

        Returns:
            Command output (stdout + stderr), or pauses for human approval.
        """
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

        parts: list[str] = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"STDERR: {result.stderr}")
        if result.exit_code != 0:
            parts.append(f"EXIT_CODE: {result.exit_code}")
        return "\n".join(parts) if parts else "(no output)"

    return shell


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


def _make_web_fetch_tool(sources_config: SourcesConfig) -> Any:
    """Return a LangChain tool that fetches web content from allowed domains.

    The tool checks the URL's domain against ``sources.json`` allowed_domains
    before making the request.
    """

    @tool
    async def web_fetch(url: str) -> str:
        """Fetch content from a URL.

        Only URLs whose domain is in the allowed_domains list (sources.json)
        can be accessed. Use this to read GitHub issues, pull requests,
        documentation pages, and other web resources.

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

        if not sources_config.is_domain_allowed(domain):
            return (
                f"Error: domain '{domain}' is not in the allowed domains list. "
                f"Check sources.json web_access.allowed_domains."
            )

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
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(
    workspace_path: str,
    permission_checker: PermissionChecker,
    sources_config: SourcesConfig,
    checkpointer: Optional[Any] = None,
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
    llm = ChatOpenAI(
        model=config.llm_model,
        base_url=config.llm_api_base,
        api_key=config.llm_api_key,
    )

    # -- Tools --------------------------------------------------------------
    tools = [
        _make_shell_tool(executor),
        _make_file_read_tool(workspace_path),
        _make_file_write_tool(workspace_path),
        _make_web_fetch_tool(sources_config),
        make_explore_tool(workspace_path, llm),   # C20: in-process sub-agent
        make_delegate_tool(),                      # C20: out-of-process sub-agent
    ]

    llm_with_tools = llm.bind_tools(tools)

    # -- Graph nodes --------------------------------------------------------

    async def assistant(state: SandboxState) -> dict[str, Any]:
        """Invoke the LLM with the current messages."""
        system = SystemMessage(content=_SYSTEM_PROMPT)
        messages = [system] + state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # -- Assemble graph -----------------------------------------------------
    graph = StateGraph(SandboxState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("assistant")
    # TODO(HITL): To add human-in-the-loop approval for dangerous commands:
    # 1. Add a "hitl_check" node between assistant and tools
    # 2. hitl_check inspects tool_calls for commands that need approval
    # 3. If approval needed, call interrupt({"command": cmd, "reason": reason})
    # 4. LangGraph pauses the graph until resume() is called with the decision
    # 5. The A2A task status shows "input-required" state
    # 6. Frontend shows approval buttons; user clicks approve/deny
    # 7. Backend calls resume() on the graph, execution continues
    #
    # Current implementation: interrupt() is called inside _make_shell_tool
    # (in the tool itself) when HitlRequired is raised. A graph-level
    # hitl_check node would give more control (e.g. batch approvals,
    # richer context) but requires restructuring the conditional edges:
    #   assistant -> hitl_check -> tools -> assistant
    # instead of the current:
    #   assistant -> tools -> assistant
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile(checkpointer=checkpointer)
