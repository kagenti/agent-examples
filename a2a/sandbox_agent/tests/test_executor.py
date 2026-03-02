"""Tests for the sandbox executor.

Validates that the SandboxExecutor:
  - Checks permissions before running any command
  - Returns an error ExecutionResult for denied commands
  - Raises HitlRequired for unknown commands (HITL)
  - Executes allowed commands in the workspace directory
  - Enforces timeout from SourcesConfig
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile

import pytest

from sandbox_agent.executor import ExecutionResult, HitlRequired, SandboxExecutor
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
    """A SourcesConfig with a short timeout for testing."""
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
    """Create a temporary workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture()
def executor(
    workspace: pathlib.Path,
    checker: PermissionChecker,
    sources_config: SourcesConfig,
) -> SandboxExecutor:
    return SandboxExecutor(
        workspace_path=str(workspace),
        permission_checker=checker,
        sources_config=sources_config,
    )


# ---------------------------------------------------------------------------
# Allowed commands
# ---------------------------------------------------------------------------


class TestAllowedCommands:
    """Commands in the allow list should execute and return output."""

    @pytest.mark.asyncio
    async def test_grep_runs_and_returns_output(
        self, executor: SandboxExecutor, workspace: pathlib.Path
    ) -> None:
        """grep is allowed -- should run and produce stdout."""
        # Create a file to grep
        test_file = workspace / "hello.txt"
        test_file.write_text("hello world\ngoodbye world\n")

        result = await executor.run_shell("grep hello hello.txt")

        assert isinstance(result, ExecutionResult)
        assert result.exit_code == 0
        assert "hello world" in result.stdout

    @pytest.mark.asyncio
    async def test_ls_shows_workspace_contents(
        self, executor: SandboxExecutor, workspace: pathlib.Path
    ) -> None:
        """ls is allowed -- should list workspace files."""
        (workspace / "file_a.txt").write_text("a")
        (workspace / "file_b.txt").write_text("b")

        result = await executor.run_shell("ls")

        assert result.exit_code == 0
        assert "file_a.txt" in result.stdout
        assert "file_b.txt" in result.stdout

    @pytest.mark.asyncio
    async def test_write_and_read_script(
        self, executor: SandboxExecutor, workspace: pathlib.Path
    ) -> None:
        """echo to file then bash execute -- both are allowed."""
        # Write a script using echo (allowed)
        write_result = await executor.run_shell(
            'echo \'#!/bin/bash\necho "script ran"\' > myscript.sh'
        )
        assert write_result.exit_code == 0

        # Execute the script using bash (allowed)
        run_result = await executor.run_shell("bash myscript.sh")
        assert run_result.exit_code == 0
        assert "script ran" in run_result.stdout


# ---------------------------------------------------------------------------
# Denied commands
# ---------------------------------------------------------------------------


class TestDeniedCommands:
    """Commands in the deny list should return an error ExecutionResult."""

    @pytest.mark.asyncio
    async def test_curl_denied(self, executor: SandboxExecutor) -> None:
        """curl is in the deny list -- should return error result."""
        result = await executor.run_shell("curl https://example.com")

        assert isinstance(result, ExecutionResult)
        assert result.exit_code != 0
        assert "denied" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_sudo_denied(self, executor: SandboxExecutor) -> None:
        """sudo is in the deny list -- should return error result."""
        result = await executor.run_shell("sudo ls")

        assert isinstance(result, ExecutionResult)
        assert result.exit_code != 0
        assert "denied" in result.stderr.lower()


# ---------------------------------------------------------------------------
# HITL (unknown commands)
# ---------------------------------------------------------------------------


class TestHitlCommands:
    """Commands not in allow or deny should raise HitlRequired."""

    @pytest.mark.asyncio
    async def test_docker_raises_hitl(self, executor: SandboxExecutor) -> None:
        """docker is not in allow or deny -- should raise HitlRequired."""
        with pytest.raises(HitlRequired) as exc_info:
            await executor.run_shell("docker run alpine")

        assert exc_info.value.command == "docker run alpine"

    @pytest.mark.asyncio
    async def test_unknown_command_raises_hitl(
        self, executor: SandboxExecutor
    ) -> None:
        """A completely unknown command should raise HitlRequired."""
        with pytest.raises(HitlRequired) as exc_info:
            await executor.run_shell("some_random_binary --flag")

        assert exc_info.value.command == "some_random_binary --flag"


# ---------------------------------------------------------------------------
# Timeout enforcement
# ---------------------------------------------------------------------------


class TestTimeout:
    """Commands exceeding the timeout should be killed."""

    @pytest.mark.asyncio
    async def test_timeout_kills_long_running_command(
        self, workspace: pathlib.Path, checker: PermissionChecker
    ) -> None:
        """sleep 30 with a 2s timeout should be killed."""
        short_timeout_config = SourcesConfig.from_dict(
            {
                "runtime": {
                    "max_execution_time_seconds": 2,
                    "max_memory_mb": 512,
                }
            }
        )
        executor = SandboxExecutor(
            workspace_path=str(workspace),
            permission_checker=checker,
            sources_config=short_timeout_config,
        )

        # bash is allowed; sleep 30 should be killed after 2s
        result = await executor.run_shell("bash -c 'sleep 30'")

        assert result.exit_code != 0
        assert "timeout" in result.stderr.lower() or "timed out" in result.stderr.lower()


# ---------------------------------------------------------------------------
# ExecutionResult dataclass
# ---------------------------------------------------------------------------


class TestExecutionResult:
    """Basic smoke tests for the ExecutionResult dataclass."""

    def test_fields(self) -> None:
        r = ExecutionResult(stdout="out", stderr="err", exit_code=0)
        assert r.stdout == "out"
        assert r.stderr == "err"
        assert r.exit_code == 0


# ---------------------------------------------------------------------------
# HitlRequired exception
# ---------------------------------------------------------------------------


class TestHitlRequiredException:
    """Basic tests for HitlRequired."""

    def test_has_command_attribute(self) -> None:
        exc = HitlRequired("git push origin main")
        assert exc.command == "git push origin main"

    def test_is_exception(self) -> None:
        assert issubclass(HitlRequired, Exception)
