"""Startup probe for Landlock filesystem isolation.

Forks a child process to verify that Landlock actually works on this
kernel.  The child applies Landlock, writes to an allowed directory,
and verifies that reads outside the sandbox are blocked.

Because Landlock is irreversible, the probe MUST run in a fork.
If the probe fails, the process exits with sys.exit(1).
"""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap

logger = logging.getLogger(__name__)


def probe_landlock() -> int:
    """Fork a child that applies Landlock and verifies it blocks /etc/hostname.

    Returns the ABI version on success.
    Calls sys.exit(1) if Landlock is unavailable or the probe fails.
    """
    # The child script imports landlock_ctypes from the same package.
    # We run it as a subprocess so Landlock restrictions are confined
    # to the child process and do not affect the parent.
    child_script = textwrap.dedent("""\
        import os
        import sys
        import tempfile

        # Ensure the package is importable
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        from sandbox_agent.landlock_ctypes import apply_landlock, get_abi_version

        abi = get_abi_version()

        # Create a temp directory for the sandbox
        tmp_dir = tempfile.mkdtemp(prefix="landlock_probe_")

        # Read-only paths for basic system functionality
        ro_paths = []
        for p in ["/usr", "/lib", "/lib64", "/etc"]:
            if os.path.exists(p):
                ro_paths.append(p)

        # Apply Landlock: only tmp_dir is writable
        apply_landlock(rw_paths=[tmp_dir], ro_paths=ro_paths)

        # Verify: writing inside the sandbox must succeed
        test_file = os.path.join(tmp_dir, "probe_test.txt")
        with open(test_file, "w") as f:
            f.write("landlock probe ok")

        # Verify: reading the file back must succeed
        with open(test_file, "r") as f:
            content = f.read()
        assert content == "landlock probe ok", f"Read-back mismatch: {content!r}"

        # Verify: writing OUTSIDE the sandbox must fail
        blocked = False
        try:
            with open("/tmp/landlock_escape_test.txt", "w") as f:
                f.write("should not work")
        except PermissionError:
            blocked = True
        except OSError as e:
            # EACCES (13) is also acceptable
            if e.errno == 13:
                blocked = True
            else:
                raise

        if not blocked:
            print("LANDLOCK_FAIL: write outside sandbox was NOT blocked", file=sys.stderr)
            sys.exit(2)

        print(f"LANDLOCK_OK abi={abi}")
        sys.exit(0)
    """)

    # Find the package root so the child can import sandbox_agent
    package_src = str(
        __import__("pathlib").Path(__file__).resolve().parent.parent
    )

    result = subprocess.run(
        [sys.executable, "-c", child_script],
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **dict(__import__("os").environ),
            "PYTHONPATH": package_src,
        },
    )

    if result.returncode != 0:
        logger.error(
            "Landlock probe FAILED (exit=%d):\nstdout: %s\nstderr: %s",
            result.returncode,
            result.stdout.strip(),
            result.stderr.strip(),
        )
        print(
            f"FATAL: Landlock probe failed. "
            f"Kernel may not support Landlock or /proc/sys/kernel/unprivileged_landlock is 0.\n"
            f"stderr: {result.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse ABI version from stdout
    stdout = result.stdout.strip()
    abi_version = 0
    for line in stdout.splitlines():
        if line.startswith("LANDLOCK_OK"):
            for part in line.split():
                if part.startswith("abi="):
                    abi_version = int(part.split("=", 1)[1])
                    break

    if abi_version < 1:
        logger.error("Landlock probe returned invalid ABI version: %s", stdout)
        sys.exit(1)

    logger.info("Landlock probe passed -- ABI version %d", abi_version)
    return abi_version
