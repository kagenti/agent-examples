"""Sandbox executor -- runs shell commands inside a context workspace.

Every command is checked against the :class:`PermissionChecker` before
execution.  The three possible outcomes are:

  DENY  -- an error :class:`ExecutionResult` is returned immediately
  HITL  -- :class:`HitlRequired` is raised so the LangGraph graph can
           trigger an ``interrupt()`` for human approval
  ALLOW -- the command is executed via ``asyncio.create_subprocess_shell``
           inside *workspace_path* with a timeout from :class:`SourcesConfig`
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from sandbox_agent.permissions import PermissionChecker, PermissionResult
from sandbox_agent.sources import SourcesConfig


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class HitlRequired(Exception):
    """Raised when an operation needs human approval.

    Attributes
    ----------
    command:
        The shell command that requires approval.
    """

    def __init__(self, command: str) -> None:
        self.command = command
        super().__init__(f"Human approval required for command: {command}")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Captures the outcome of a shell command execution."""

    stdout: str
    stderr: str
    exit_code: int


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class SandboxExecutor:
    """Runs shell commands in a workspace directory with permission checks.

    Parameters
    ----------
    workspace_path:
        Absolute path to the workspace directory where commands execute.
    permission_checker:
        A :class:`PermissionChecker` instance for evaluating operations.
    sources_config:
        A :class:`SourcesConfig` instance providing runtime limits.
    """

    def __init__(
        self,
        workspace_path: str,
        permission_checker: PermissionChecker,
        sources_config: SourcesConfig,
    ) -> None:
        self._workspace_path = workspace_path
        self._permission_checker = permission_checker
        self._sources_config = sources_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_shell(self, command: str) -> ExecutionResult:
        """Run a shell command after checking permissions.

        Parameters
        ----------
        command:
            The shell command string to execute.

        Returns
        -------
        ExecutionResult
            On success (ALLOW) or on DENY (with a non-zero exit code and
            an error message in stderr).

        Raises
        ------
        HitlRequired
            When the command matches neither allow nor deny rules and
            requires human approval.
        """
        # 1. Extract the command prefix for permission matching.
        #    Try "cmd subcmd" first (e.g. "pip install"), then fall back
        #    to just "cmd" (e.g. "grep").
        operation = command.strip()
        permission = self._check_permission(operation)

        # 2. Act on the permission result.
        if permission is PermissionResult.DENY:
            return ExecutionResult(
                stdout="",
                stderr=f"Permission denied: command '{command}' is denied by policy.",
                exit_code=1,
            )

        if permission is PermissionResult.HITL:
            raise HitlRequired(command)

        # 3. ALLOW -- execute the command.
        return await self._execute(command)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_permission(self, operation: str) -> PermissionResult:
        """Check the permission for a shell operation.

        The permission checker expects the full command string as the
        operation.  It internally handles prefix matching (e.g. matching
        "grep -r foo" against the rule ``shell(grep:*)``).
        """
        return self._permission_checker.check("shell", operation)

    async def _execute(self, command: str) -> ExecutionResult:
        """Execute *command* in the workspace directory with a timeout."""
        timeout = self._sources_config.max_execution_time_seconds

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=self._workspace_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                # Kill the process and its children.
                try:
                    process.kill()
                except ProcessLookupError:
                    pass  # already exited
                # Wait for the process to be reaped.
                await process.wait()
                return ExecutionResult(
                    stdout="",
                    stderr=(
                        f"Command timed out after {timeout} seconds "
                        f"and was killed: '{command}'"
                    ),
                    exit_code=-1,
                )

            return ExecutionResult(
                stdout=(stdout_bytes or b"").decode("utf-8", errors="replace"),
                stderr=(stderr_bytes or b"").decode("utf-8", errors="replace"),
                exit_code=process.returncode if process.returncode is not None else -1,
            )

        except OSError as exc:
            return ExecutionResult(
                stdout="",
                stderr=f"Failed to start command: {exc}",
                exit_code=-1,
            )
