"""Raw ctypes wrapper for Linux Landlock LSM syscalls.

Architecture-aware: supports x86_64 and aarch64 syscall numbers.
Zero external dependencies -- pure ctypes + stdlib.

Landlock is IRREVERSIBLE once applied to a thread. There is no undo.
All functions in this module fail hard (raise OSError) on error.
"""

from __future__ import annotations

import ctypes
import os
import platform
import struct

# ---------------------------------------------------------------------------
# Syscall numbers by architecture
# ---------------------------------------------------------------------------

_ARCH = platform.machine()

if _ARCH == "x86_64":
    _SYS_LANDLOCK_CREATE_RULESET = 444
    _SYS_LANDLOCK_ADD_RULE = 445
    _SYS_LANDLOCK_RESTRICT_SELF = 446
elif _ARCH == "aarch64":
    _SYS_LANDLOCK_CREATE_RULESET = 441
    _SYS_LANDLOCK_ADD_RULE = 442
    _SYS_LANDLOCK_RESTRICT_SELF = 443
else:
    raise RuntimeError(f"Unsupported architecture for Landlock: {_ARCH}")

# ---------------------------------------------------------------------------
# Landlock constants
# ---------------------------------------------------------------------------

LANDLOCK_RULE_PATH_BENEATH = 1

# ABI v1 access flags (13 flags)
_ACCESS_FS_V1 = (
    (1 << 0)   # EXECUTE
    | (1 << 1)   # WRITE_FILE
    | (1 << 2)   # READ_FILE
    | (1 << 3)   # READ_DIR
    | (1 << 4)   # REMOVE_DIR
    | (1 << 5)   # REMOVE_FILE
    | (1 << 6)   # MAKE_CHAR
    | (1 << 7)   # MAKE_DIR
    | (1 << 8)   # MAKE_REG
    | (1 << 9)   # MAKE_SOCK
    | (1 << 10)  # MAKE_FIFO
    | (1 << 11)  # MAKE_BLOCK
    | (1 << 12)  # MAKE_SYM
)

# ABI v2 adds REFER
_ACCESS_FS_REFER = 1 << 13

# ABI v3 adds TRUNCATE
_ACCESS_FS_TRUNCATE = 1 << 14

# Read-only subset (for ro_paths)
ACCESS_FS_READ_ONLY = (
    (1 << 0)   # EXECUTE
    | (1 << 2)   # READ_FILE
    | (1 << 3)   # READ_DIR
)

_libc = ctypes.CDLL("libc.so.6", use_errno=True)

# ---------------------------------------------------------------------------
# Syscall helpers
# ---------------------------------------------------------------------------


def _syscall(nr: int, *args: int) -> int:
    """Invoke a raw syscall. Returns the result or raises OSError."""
    result = _libc.syscall(ctypes.c_long(nr), *[ctypes.c_long(a) for a in args])
    if result < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"syscall {nr} failed: {os.strerror(errno)}")
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_abi_version() -> int:
    """Query the kernel's Landlock ABI version.

    Returns an integer >= 1 if Landlock is supported.
    Raises OSError if Landlock is not available.
    """
    # landlock_create_ruleset(NULL, 0, LANDLOCK_CREATE_RULESET_VERSION=1<<0)
    LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
    return _syscall(_SYS_LANDLOCK_CREATE_RULESET, 0, 0, LANDLOCK_CREATE_RULESET_VERSION)


def _get_fs_access_flags(abi_version: int) -> int:
    """Return the full set of handled_access_fs flags for the given ABI version."""
    flags = _ACCESS_FS_V1
    if abi_version >= 2:
        flags |= _ACCESS_FS_REFER
    if abi_version >= 3:
        flags |= _ACCESS_FS_TRUNCATE
    return flags


def _add_rule(ruleset_fd: int, path: str, access: int) -> None:
    """Add a path-beneath rule to an existing Landlock ruleset.

    Parameters
    ----------
    ruleset_fd:
        File descriptor of the Landlock ruleset.
    path:
        Absolute filesystem path to allow.
    access:
        Bitmask of allowed access rights.
    """
    parent_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        # struct landlock_path_beneath_attr {
        #     __u64 allowed_access;   // 8 bytes
        #     __s32 parent_fd;        // 4 bytes
        #     // 4 bytes padding
        # }
        attr = struct.pack("QiI", access, parent_fd, 0)
        attr_ptr = ctypes.c_char_p(attr)
        _syscall(
            _SYS_LANDLOCK_ADD_RULE,
            ruleset_fd,
            LANDLOCK_RULE_PATH_BENEATH,
            ctypes.cast(attr_ptr, ctypes.c_void_p).value,
            0,
        )
    finally:
        os.close(parent_fd)


def apply_landlock(rw_paths: list[str], ro_paths: list[str]) -> None:
    """Create a Landlock ruleset, add path rules, and restrict the current thread.

    This is IRREVERSIBLE. After this call, the thread can only access
    the specified paths with the specified permissions.

    Parameters
    ----------
    rw_paths:
        Paths to allow full read-write access.
    ro_paths:
        Paths to allow read-only access (execute + read_file + read_dir).

    Raises
    ------
    OSError
        If any Landlock syscall fails. No fallback, no degraded mode.
    """
    abi = get_abi_version()
    handled_access_fs = _get_fs_access_flags(abi)

    # struct landlock_ruleset_attr { __u64 handled_access_fs; }
    ruleset_attr = struct.pack("Q", handled_access_fs)
    ruleset_attr_ptr = ctypes.c_char_p(ruleset_attr)
    ruleset_fd = _syscall(
        _SYS_LANDLOCK_CREATE_RULESET,
        ctypes.cast(ruleset_attr_ptr, ctypes.c_void_p).value,
        len(ruleset_attr),
        0,
    )

    try:
        # Add read-write path rules
        for path in rw_paths:
            if os.path.exists(path):
                _add_rule(ruleset_fd, path, handled_access_fs)

        # Add read-only path rules
        for path in ro_paths:
            if os.path.exists(path):
                _add_rule(ruleset_fd, path, ACCESS_FS_READ_ONLY)

        # prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) -- required before restrict_self
        PR_SET_NO_NEW_PRIVS = 38
        _libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)

        # landlock_restrict_self(ruleset_fd, 0)
        _syscall(_SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0)
    finally:
        os.close(ruleset_fd)
