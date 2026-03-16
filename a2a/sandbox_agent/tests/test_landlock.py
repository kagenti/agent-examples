"""Unit tests for Landlock filesystem isolation.

Each test runs inside a subprocess because Landlock is IRREVERSIBLE --
once applied to a thread, it cannot be removed.  We fork a child process
for each test, apply Landlock there, and check the result from the parent.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import textwrap

import pytest

# All tests require Linux with Landlock support
_IS_LINUX = sys.platform == "linux"
_ARCH = platform.machine()
_SUPPORTED_ARCH = _ARCH in ("x86_64", "aarch64")


def _run_child(script: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a Python script in a subprocess with sandbox_agent importable."""
    package_src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        "src",
    )
    env = {**os.environ, "PYTHONPATH": package_src}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


@pytest.mark.skipif(not _IS_LINUX, reason="Landlock is Linux-only")
@pytest.mark.skipif(not _SUPPORTED_ARCH, reason=f"Unsupported arch: {_ARCH}")
class TestLandlockCtypes:
    """Tests for landlock_ctypes module -- each runs in a subprocess."""

    def test_abi_version_detection(self):
        """get_abi_version() should return an int >= 1."""
        result = _run_child("""\
            from sandbox_agent.landlock_ctypes import get_abi_version
            abi = get_abi_version()
            assert isinstance(abi, int), f"Expected int, got {type(abi)}"
            assert abi >= 1, f"Expected ABI >= 1, got {abi}"
            print(f"ABI={abi}")
        """)
        assert result.returncode == 0, (
            f"Child failed (exit={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "ABI=" in result.stdout

    def test_apply_landlock_allows_workspace(self):
        """After apply_landlock, writing to an allowed directory must succeed."""
        result = _run_child("""\
            import os
            import tempfile
            from sandbox_agent.landlock_ctypes import apply_landlock

            # Create workspace
            ws = tempfile.mkdtemp(prefix="ll_test_ws_")

            # Read-only system paths
            ro = [p for p in ["/usr", "/lib", "/lib64", "/etc", "/proc"]
                  if os.path.exists(p)]

            apply_landlock(rw_paths=[ws], ro_paths=ro)

            # Write must succeed
            test_file = os.path.join(ws, "test.txt")
            with open(test_file, "w") as f:
                f.write("hello landlock")

            with open(test_file, "r") as f:
                content = f.read()

            assert content == "hello landlock", f"Content mismatch: {content!r}"
            print("WRITE_OK")
        """)
        assert result.returncode == 0, (
            f"Child failed (exit={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "WRITE_OK" in result.stdout

    def test_apply_landlock_blocks_outside(self):
        """After apply_landlock, reading /etc/hostname must raise PermissionError."""
        result = _run_child("""\
            import os
            import tempfile
            from sandbox_agent.landlock_ctypes import apply_landlock

            ws = tempfile.mkdtemp(prefix="ll_test_block_")

            # Only allow the workspace -- NO /etc in ro_paths
            apply_landlock(rw_paths=[ws], ro_paths=[])

            # Attempt to read /etc/hostname should fail
            blocked = False
            try:
                with open("/etc/hostname", "r") as f:
                    f.read()
            except PermissionError:
                blocked = True
            except OSError as e:
                if e.errno == 13:  # EACCES
                    blocked = True
                else:
                    raise

            assert blocked, "Reading /etc/hostname was NOT blocked!"
            print("BLOCK_OK")
        """)
        assert result.returncode == 0, (
            f"Child failed (exit={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "BLOCK_OK" in result.stdout

    def test_architecture_detection(self):
        """Syscall numbers must be correct for the current platform.machine()."""
        result = _run_child("""\
            import platform
            from sandbox_agent import landlock_ctypes as ll

            arch = platform.machine()
            if arch == "x86_64":
                assert ll._SYS_LANDLOCK_CREATE_RULESET == 444
                assert ll._SYS_LANDLOCK_ADD_RULE == 445
                assert ll._SYS_LANDLOCK_RESTRICT_SELF == 446
            elif arch == "aarch64":
                assert ll._SYS_LANDLOCK_CREATE_RULESET == 441
                assert ll._SYS_LANDLOCK_ADD_RULE == 442
                assert ll._SYS_LANDLOCK_RESTRICT_SELF == 443
            else:
                raise AssertionError(f"Unexpected arch: {arch}")
            print(f"ARCH_OK={arch}")
        """)
        assert result.returncode == 0, (
            f"Child failed (exit={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "ARCH_OK=" in result.stdout


@pytest.mark.skipif(not _IS_LINUX, reason="Landlock is Linux-only")
@pytest.mark.skipif(not _SUPPORTED_ARCH, reason=f"Unsupported arch: {_ARCH}")
class TestLandlockProbe:
    """Tests for the landlock_probe module."""

    def test_probe_passes(self):
        """probe_landlock() should return ABI version without exiting."""
        result = _run_child("""\
            from sandbox_agent.landlock_probe import probe_landlock
            abi = probe_landlock()
            assert isinstance(abi, int), f"Expected int, got {type(abi)}"
            assert abi >= 1, f"Expected ABI >= 1, got {abi}"
            print(f"PROBE_OK abi={abi}")
        """)
        assert result.returncode == 0, (
            f"Probe failed (exit={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "PROBE_OK" in result.stdout
