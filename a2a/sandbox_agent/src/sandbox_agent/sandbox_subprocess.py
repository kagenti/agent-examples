"""Per-tool-call Landlock isolation via subprocess fork.

Each command execution forks a child process that applies Landlock
restrictions before executing the command. This ensures that even
if the command is malicious, it cannot escape the workspace.

The Landlock restrictions are:
- rw_paths: workspace directory + session-specific /tmp
- ro_paths: system directories needed for basic command execution

There is NO fallback. If Landlock fails, the subprocess fails.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum output size to capture (prevent OOM on runaway commands)
_MAX_OUTPUT_BYTES = 10 * 1024 * 1024  # 10 MB


async def sandboxed_subprocess(
    command: str,
    workspace_path: str,
    timeout: float = 120.0,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Execute a command inside a Landlock-restricted subprocess.

    Forks a child process that:
    1. Applies Landlock restricting filesystem access to workspace + system dirs
    2. Executes the command via shell

    Parameters
    ----------
    command:
        Shell command string to execute.
    workspace_path:
        Absolute path to the session workspace (read-write).
    timeout:
        Maximum execution time in seconds.
    env:
        Optional extra environment variables for the child.

    Returns
    -------
    tuple[int, str, str]
        (returncode, stdout, stderr)

    Raises
    ------
    OSError
        If Landlock application fails in the child (propagated via non-zero exit).
    """
    # Create session-specific tmp directory
    # Use a hash of workspace_path to create a unique tmp dir
    ws_hash = hashlib.sha256(workspace_path.encode()).hexdigest()[:12]
    session_tmp = f"/tmp/sandbox_{ws_hash}"
    Path(session_tmp).mkdir(parents=True, exist_ok=True)

    # Build the child script that applies Landlock then execs the command
    # The child script is passed via -c to the Python interpreter
    child_script = textwrap.dedent("""\
        import os
        import subprocess
        import sys

        # Import the landlock module from the package
        sys.path.insert(0, os.environ["_LANDLOCK_PYTHONPATH"])
        from sandbox_agent.landlock_ctypes import apply_landlock

        workspace = os.environ["SANDBOX_WORKSPACE"]
        session_tmp = os.environ["SANDBOX_TMP"]

        # Collect read-only system paths that exist
        ro_paths = []
        for p in ["/usr", "/bin", "/lib", "/lib64", "/opt", "/etc",
                  "/proc", "/dev/null", "/dev/urandom", "/app"]:
            if os.path.exists(p):
                ro_paths.append(p)

        # Add Python prefix for stdlib access
        prefix = sys.prefix
        if os.path.exists(prefix) and prefix not in ro_paths:
            ro_paths.append(prefix)

        # Apply Landlock -- NO try/except, hard fail if this fails
        apply_landlock(
            rw_paths=[workspace, session_tmp],
            ro_paths=ro_paths,
        )

        # Execute the user command
        result = subprocess.run(
            os.environ["_LANDLOCK_COMMAND"],
            shell=True,
            cwd=workspace,
            capture_output=True,
            timeout=float(os.environ.get("_LANDLOCK_TIMEOUT", "120")),
        )

        # Write stdout and stderr to fds 1 and 2
        sys.stdout.buffer.write(result.stdout)
        sys.stderr.buffer.write(result.stderr)
        sys.exit(result.returncode)
    """)

    # Build environment for the child process
    child_env = dict(os.environ)
    if env:
        child_env.update(env)

    # Find package source directory for PYTHONPATH
    package_src = str(Path(__file__).resolve().parent.parent)

    child_env["SANDBOX_WORKSPACE"] = workspace_path
    child_env["SANDBOX_TMP"] = session_tmp
    child_env["_LANDLOCK_PYTHONPATH"] = package_src
    child_env["_LANDLOCK_COMMAND"] = command
    child_env["_LANDLOCK_TIMEOUT"] = str(timeout)

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", child_script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env,
            cwd=workspace_path,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout + 5,  # extra margin for Landlock setup
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
            return (
                -1,
                "",
                f"Sandboxed command timed out after {timeout} seconds: '{command}'",
            )

        stdout = (stdout_bytes or b"")[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        stderr = (stderr_bytes or b"")[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        returncode = process.returncode if process.returncode is not None else -1

        return (returncode, stdout, stderr)

    except OSError as exc:
        return (-1, "", f"Failed to start sandboxed subprocess: {exc}")
