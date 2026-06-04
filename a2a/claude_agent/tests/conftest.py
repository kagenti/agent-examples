import os
import stat

import pytest

# An NDJSON stream that mimics one Claude turn: init → assistant text →
# tool_use → tool_result → success result.
_FAKE_STREAM = "\n".join(
    [
        '{"type":"system","subtype":"init","session_id":"SID"}',
        '{"type":"assistant","message":{"content":[{"type":"text","text":"working on it"}]}}',
        '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Bash","input":{"command":"echo hi"}}]}}',
        '{"type":"user","message":{"content":[{"type":"tool_result","content":"hi"}]}}',
        '{"type":"result","subtype":"success","is_error":false,"result":"all done","session_id":"SID"}',
    ]
)


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    """Put a fake `claude` executable on PATH that prints a canned NDJSON stream."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "claude"
    script.write_text(f"#!/usr/bin/env bash\ncat <<'EOF'\n{_FAKE_STREAM}\nEOF\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return script
