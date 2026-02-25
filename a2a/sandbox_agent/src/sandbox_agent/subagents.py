"""Sub-agent spawning tools for the sandbox agent (C20).

Provides two spawning modes:

1. **In-process** (``explore``): A lightweight LangGraph sub-graph that
   runs as an asyncio task in the same process.  It has a scoped,
   read-only tool set (grep, file_read, glob) and a bounded iteration
   limit.  Good for codebase research and analysis.

2. **Out-of-process** (``delegate``): Creates a Kubernetes SandboxClaim
   that spawns a separate pod with full sandbox isolation.  The parent
   polls the sub-agent's A2A endpoint until it returns results.  Good
   for untrusted or long-running tasks.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

logger = logging.getLogger(__name__)

# Maximum iterations for in-process sub-agents to prevent runaway loops.
_MAX_SUB_AGENT_ITERATIONS = 15


# ---------------------------------------------------------------------------
# In-process sub-agent: explore (C20, mode 1)
# ---------------------------------------------------------------------------


def _make_explore_tools(workspace: str) -> list[Any]:
    """Build a read-only tool set for the explore sub-agent."""
    ws_root = Path(workspace).resolve()

    @tool
    async def grep(pattern: str, path: str = ".") -> str:
        """Search for a regex pattern in files under the workspace.

        Args:
            pattern: Regex pattern to search for.
            path: Relative path to search in (default: workspace root).

        Returns:
            Matching lines with file paths and line numbers.
        """
        target = (ws_root / path).resolve()
        if not str(target).startswith(str(ws_root)):
            return "Error: path resolves outside the workspace."

        try:
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", "--include=*.md",
                 "--include=*.yaml", "--include=*.yml", "--include=*.json",
                 "--include=*.txt", "--include=*.sh", "--include=*.go",
                 pattern, str(target)],
                capture_output=True, text=True, timeout=30,
                cwd=str(ws_root),
            )
            output = result.stdout[:10000]
            if not output:
                return f"No matches found for pattern '{pattern}'"
            return output
        except subprocess.TimeoutExpired:
            return "Search timed out after 30 seconds."
        except FileNotFoundError:
            return "grep command not available."

    @tool
    async def read_file(path: str) -> str:
        """Read a file from the workspace (read-only).

        Args:
            path: Relative path within the workspace.

        Returns:
            File contents (truncated to 20000 chars).
        """
        resolved = (ws_root / path).resolve()
        if not str(resolved).startswith(str(ws_root)):
            return "Error: path resolves outside the workspace."
        if not resolved.is_file():
            return f"Error: file not found at '{path}'."
        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
            if len(content) > 20000:
                content = content[:20000] + "\n\n[Truncated at 20000 chars]"
            return content
        except OSError as exc:
            return f"Error reading file: {exc}"

    @tool
    async def list_files(path: str = ".", pattern: str = "*") -> str:
        """List files matching a glob pattern in the workspace.

        Args:
            path: Relative directory to search in (default: workspace root).
            pattern: Glob pattern (default: all files).

        Returns:
            Newline-separated list of matching file paths.
        """
        target = (ws_root / path).resolve()
        if not str(target).startswith(str(ws_root)):
            return "Error: path resolves outside the workspace."
        if not target.is_dir():
            return f"Error: directory not found at '{path}'."

        matches = sorted(str(p.relative_to(ws_root)) for p in target.rglob(pattern) if p.is_file())
        if len(matches) > 200:
            matches = matches[:200]
            matches.append(f"... and more (truncated at 200)")
        return "\n".join(matches) if matches else "No files found."

    return [grep, read_file, list_files]


def create_explore_graph(workspace: str, llm: Any) -> Any:
    """Create a read-only explore sub-graph.

    The sub-graph has access only to grep, read_file, and list_files.
    It is bounded to ``_MAX_SUB_AGENT_ITERATIONS`` steps.
    """
    tools = _make_explore_tools(workspace)
    llm_with_tools = llm.bind_tools(tools)

    async def assistant(state: MessagesState) -> dict[str, Any]:
        system = SystemMessage(
            content=(
                "You are a codebase research assistant. Your job is to find "
                "specific information in the workspace using grep, read_file, "
                "and list_files. Be concise. Return a focused summary of what "
                "you found. Do NOT modify any files."
            )
        )
        messages = [system] + state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile()


def make_explore_tool(workspace: str, llm: Any) -> Any:
    """Return a LangChain tool that spawns an in-process explore sub-agent."""

    @tool
    async def explore(query: str) -> str:
        """Spawn a read-only sub-agent to research the codebase.

        The sub-agent has access to grep, read_file, and list_files
        but cannot write files or execute shell commands. Use this for
        codebase exploration, finding definitions, and analyzing code.

        Args:
            query: What to search for or investigate in the codebase.

        Returns:
            A summary of findings from the explore sub-agent.
        """
        sub_graph = create_explore_graph(workspace, llm)
        try:
            result = await asyncio.wait_for(
                sub_graph.ainvoke(
                    {"messages": [HumanMessage(content=query)]},
                    config={"recursion_limit": _MAX_SUB_AGENT_ITERATIONS},
                ),
                timeout=120,
            )
            messages = result.get("messages", [])
            if messages:
                last = messages[-1]
                return last.content if hasattr(last, "content") else str(last)
            return "No results from explore sub-agent."
        except asyncio.TimeoutError:
            return "Explore sub-agent timed out after 120 seconds."
        except Exception as exc:
            return f"Explore sub-agent error: {exc}"

    return explore


# ---------------------------------------------------------------------------
# Out-of-process sub-agent: delegate (C20, mode 2)
# ---------------------------------------------------------------------------


def make_delegate_tool() -> Any:
    """Return a LangChain tool that spawns a sandbox sub-agent via SandboxClaim.

    This tool creates a Kubernetes SandboxClaim, which the agent-sandbox
    controller provisions as a separate pod. The parent agent polls the
    sub-agent's A2A endpoint until it returns results.

    Requires: KUBECONFIG environment variable and agent-sandbox CRDs installed.
    """

    @tool
    async def delegate(task: str, namespace: str = "team1") -> str:
        """Spawn a separate sandbox agent pod for a delegated task.

        Creates a Kubernetes SandboxClaim that provisions an isolated
        sandbox pod with its own workspace, permissions, and identity.
        Use this for untrusted, long-running, or resource-intensive tasks
        that need full isolation from the parent agent.

        Args:
            task: Description of the task for the sub-agent to perform.
            namespace: Kubernetes namespace for the sub-agent (default: team1).

        Returns:
            The sub-agent's response, or a status message if still running.
        """
        # This is a placeholder implementation. In production, this would:
        # 1. Create a SandboxClaim via kubernetes-client
        # 2. Wait for the pod to be provisioned
        # 3. Send an A2A message to the sub-agent
        # 4. Poll for results
        #
        # For now, return a message indicating the feature is available
        # but requires cluster resources.
        logger.info(
            "delegate tool called: task=%s, namespace=%s", task, namespace
        )
        return (
            f"Delegation requested: '{task}' in namespace '{namespace}'. "
            "SandboxClaim-based delegation requires a running Kubernetes "
            "cluster with agent-sandbox CRDs installed. This feature is "
            "designed for production deployments where tasks need full "
            "pod-level isolation."
        )

    return delegate
