"""Tests for the LangGraph agent graph.

Validates that:
  - SandboxState has required fields (context_id, workspace_path, final_answer)
  - build_graph returns a compiled graph with an ainvoke method
  - _make_shell_tool returns a tool that delegates to executor.run_shell
  - _make_file_read_tool reads files relative to workspace and blocks traversal
  - _make_file_write_tool writes files relative to workspace and blocks traversal
"""

from __future__ import annotations

import json
import os
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph.checkpoint.memory import MemorySaver

from sandbox_agent.executor import ExecutionResult, HitlRequired
from sandbox_agent.graph import (
    SandboxState,
    _make_file_read_tool,
    _make_file_write_tool,
    _make_shell_tool,
    build_graph,
)
from sandbox_agent.permissions import PermissionChecker
from sandbox_agent.sources import SourcesConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SETTINGS_PATH = pathlib.Path(__file__).resolve().parents[1] / "settings.json"


@pytest.fixture()
def settings() -> dict:
    """Load the real settings.json shipped with the agent."""
    with open(SETTINGS_PATH) as fh:
        return json.load(fh)


@pytest.fixture()
def checker(settings: dict) -> PermissionChecker:
    return PermissionChecker(settings)


@pytest.fixture()
def sources_config() -> SourcesConfig:
    return SourcesConfig.from_dict(
        {
            "runtime": {
                "max_execution_time_seconds": 10,
                "max_memory_mb": 512,
            }
        }
    )


@pytest.fixture()
def workspace(tmp_path: pathlib.Path) -> pathlib.Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


# ---------------------------------------------------------------------------
# SandboxState
# ---------------------------------------------------------------------------


class TestSandboxState:
    """SandboxState should extend MessagesState with extra fields."""

    def test_has_context_id_annotation(self) -> None:
        annotations = SandboxState.__annotations__
        assert "context_id" in annotations

    def test_has_workspace_path_annotation(self) -> None:
        annotations = SandboxState.__annotations__
        assert "workspace_path" in annotations

    def test_has_final_answer_annotation(self) -> None:
        annotations = SandboxState.__annotations__
        assert "final_answer" in annotations


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------


class TestBuildGraph:
    """build_graph should return a compiled LangGraph with ainvoke."""

    @patch("sandbox_agent.graph.ChatOpenAI")
    def test_returns_compiled_graph(
        self,
        mock_chat_cls: MagicMock,
        workspace: pathlib.Path,
        checker: PermissionChecker,
        sources_config: SourcesConfig,
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat_cls.return_value = mock_llm

        graph = build_graph(
            workspace_path=str(workspace),
            permission_checker=checker,
            sources_config=sources_config,
        )

        assert hasattr(graph, "ainvoke"), "compiled graph must have ainvoke"

    @patch("sandbox_agent.graph.ChatOpenAI")
    def test_accepts_optional_checkpointer(
        self,
        mock_chat_cls: MagicMock,
        workspace: pathlib.Path,
        checker: PermissionChecker,
        sources_config: SourcesConfig,
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat_cls.return_value = mock_llm

        checkpointer = MemorySaver()

        graph = build_graph(
            workspace_path=str(workspace),
            permission_checker=checker,
            sources_config=sources_config,
            checkpointer=checkpointer,
        )

        assert hasattr(graph, "ainvoke")


# ---------------------------------------------------------------------------
# _make_shell_tool
# ---------------------------------------------------------------------------


class TestMakeShellTool:
    """The shell tool should delegate to executor.run_shell."""

    @pytest.mark.asyncio
    async def test_shell_tool_calls_executor(self) -> None:
        executor = AsyncMock()
        executor.run_shell.return_value = ExecutionResult(
            stdout="hello", stderr="", exit_code=0
        )

        shell_tool = _make_shell_tool(executor)
        result = await shell_tool.ainvoke({"command": "echo hello"})

        executor.run_shell.assert_awaited_once_with("echo hello")
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_shell_tool_returns_approval_on_hitl(self) -> None:
        executor = AsyncMock()
        executor.run_shell.side_effect = HitlRequired("docker run alpine")

        shell_tool = _make_shell_tool(executor)
        result = await shell_tool.ainvoke({"command": "docker run alpine"})

        assert "APPROVAL_REQUIRED" in result

    @pytest.mark.asyncio
    async def test_shell_tool_includes_stderr_on_failure(self) -> None:
        executor = AsyncMock()
        executor.run_shell.return_value = ExecutionResult(
            stdout="", stderr="Permission denied", exit_code=1
        )

        shell_tool = _make_shell_tool(executor)
        result = await shell_tool.ainvoke({"command": "curl http://example.com"})

        assert "Permission denied" in result


# ---------------------------------------------------------------------------
# _make_file_read_tool
# ---------------------------------------------------------------------------


class TestMakeFileReadTool:
    """The file_read tool should read files and prevent path traversal."""

    @pytest.mark.asyncio
    async def test_reads_file_relative_to_workspace(
        self, workspace: pathlib.Path
    ) -> None:
        (workspace / "test.txt").write_text("file contents")
        tool = _make_file_read_tool(str(workspace))
        result = await tool.ainvoke({"path": "test.txt"})
        assert "file contents" in result

    @pytest.mark.asyncio
    async def test_blocks_path_traversal(
        self, workspace: pathlib.Path
    ) -> None:
        tool = _make_file_read_tool(str(workspace))
        result = await tool.ainvoke({"path": "../../etc/passwd"})
        assert "error" in result.lower() or "denied" in result.lower() or "outside" in result.lower()

    @pytest.mark.asyncio
    async def test_missing_file_returns_error(
        self, workspace: pathlib.Path
    ) -> None:
        tool = _make_file_read_tool(str(workspace))
        result = await tool.ainvoke({"path": "nonexistent.txt"})
        assert "error" in result.lower() or "not found" in result.lower()


# ---------------------------------------------------------------------------
# _make_file_write_tool
# ---------------------------------------------------------------------------


class TestMakeFileWriteTool:
    """The file_write tool should write files and prevent path traversal."""

    @pytest.mark.asyncio
    async def test_writes_file_relative_to_workspace(
        self, workspace: pathlib.Path
    ) -> None:
        tool = _make_file_write_tool(str(workspace))
        result = await tool.ainvoke({"path": "out.txt", "content": "hello"})

        written = (workspace / "out.txt").read_text()
        assert written == "hello"
        assert "error" not in result.lower()

    @pytest.mark.asyncio
    async def test_creates_parent_dirs(
        self, workspace: pathlib.Path
    ) -> None:
        tool = _make_file_write_tool(str(workspace))
        result = await tool.ainvoke(
            {"path": "sub/dir/file.txt", "content": "nested"}
        )

        written = (workspace / "sub" / "dir" / "file.txt").read_text()
        assert written == "nested"

    @pytest.mark.asyncio
    async def test_blocks_path_traversal(
        self, workspace: pathlib.Path
    ) -> None:
        tool = _make_file_write_tool(str(workspace))
        result = await tool.ainvoke(
            {"path": "../../etc/evil", "content": "bad"}
        )
        assert "error" in result.lower() or "denied" in result.lower() or "outside" in result.lower()
        # The file must NOT have been created outside the workspace.
        assert not os.path.exists("/etc/evil")
