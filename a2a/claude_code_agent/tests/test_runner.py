import asyncio
import os
import stat
from unittest.mock import AsyncMock, MagicMock

import pytest
from claude_code_agent.configuration import Configuration
from claude_code_agent.events import StreamTranslator
from claude_code_agent.runner import build_argv, build_env, run_turn
from claude_code_agent.session import ClaudeSession


def _mk_translator():
    tu = MagicMock()
    tu.context_id, tu.task_id = "ctx-1", "task-1"
    tu.update_status = AsyncMock()
    tu.add_artifact = AsyncMock()
    tu.complete = AsyncMock()
    tu.failed = AsyncMock()
    return StreamTranslator(tu)


def _write_fake_claude(tmp_path, monkeypatch, body: str):
    """Install a fake `claude` on PATH whose script body is `body`."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    script = bindir / "claude"
    script.write_text("#!/usr/bin/env bash\n" + body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return script


def test_build_argv_first_turn_uses_session_id(tmp_path):
    s = ClaudeSession("ctx-1", str(tmp_path))
    argv = build_argv(s, "hello", "sonnet")
    assert "--session-id" in argv
    assert s.session_uuid in argv
    assert "--resume" not in argv
    assert "--dangerously-skip-permissions" in argv
    assert argv[:3] == ["claude", "-p", "hello"]


def test_build_argv_resume_after_started(tmp_path):
    s = ClaudeSession("ctx-1", str(tmp_path))
    s.started = True
    argv = build_argv(s, "again", "sonnet")
    assert "--resume" in argv
    assert "--session-id" not in argv


def test_build_env_sets_litellm_wiring(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    cfg = Configuration(_env_file=None)
    cfg.anthropic_auth_token = "sk-x"
    cfg.anthropic_base_url = "https://litellm.example.com"
    env = build_env(cfg)
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-x"
    assert env["ANTHROPIC_BASE_URL"] == "https://litellm.example.com"
    assert env["ANTHROPIC_MODEL"] == "sonnet"
    assert env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] == "1"
    assert env["HOME"] == cfg.home_dir


def test_build_env_omits_empty_base_url(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    cfg = Configuration(_env_file=None)  # base_url defaults to ""
    env = build_env(cfg)
    assert "ANTHROPIC_BASE_URL" not in env


async def test_run_turn_consumes_stream_and_completes(tmp_path, fake_claude):
    cfg = Configuration(_env_file=None)
    cfg.workspace_root = str(tmp_path / "ws")
    cfg.home_dir = str(tmp_path / "home")
    s = ClaudeSession("ctx-1", cfg.workspace_root)
    tu = MagicMock()
    tu.context_id, tu.task_id = "ctx-1", "task-1"
    tu.update_status = AsyncMock()
    tu.add_artifact = AsyncMock()
    tu.complete = AsyncMock()
    tu.failed = AsyncMock()
    translator = StreamTranslator(tu)

    await run_turn(s, "hello", translator, cfg)

    assert translator.final_text == "all done"
    assert s.started is True  # next turn will --resume
    tu.update_status.assert_awaited()  # progress was streamed


async def test_run_turn_nonzero_exit_sets_error(tmp_path, monkeypatch):
    _write_fake_claude(tmp_path, monkeypatch, "echo 'boom' >&2\nexit 1\n")
    cfg = Configuration(_env_file=None)
    cfg.workspace_root = str(tmp_path / "ws")
    cfg.home_dir = str(tmp_path / "home")
    s = ClaudeSession("ctx-1", cfg.workspace_root)
    translator = _mk_translator()

    await run_turn(s, "hello", translator, cfg)

    assert translator.errored is True
    assert "code 1" in (translator.error_reason or "")
    assert s.started is False


async def test_run_turn_timeout_kills_process(tmp_path, monkeypatch):
    pidfile = tmp_path / "pid"
    _write_fake_claude(tmp_path, monkeypatch, f"echo $$ > {pidfile}\nsleep 30\n")
    cfg = Configuration(_env_file=None)
    cfg.workspace_root = str(tmp_path / "ws")
    cfg.home_dir = str(tmp_path / "home")
    cfg.turn_timeout_s = 1
    s = ClaudeSession("ctx-1", cfg.workspace_root)
    translator = _mk_translator()

    await run_turn(s, "hello", translator, cfg)

    assert translator.errored is True
    assert "timed out" in (translator.error_reason or "")
    # The subprocess must have been killed (not left running).
    pid = int(pidfile.read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_run_turn_cancel_kills_process(tmp_path, monkeypatch):
    pidfile = tmp_path / "pid"
    _write_fake_claude(tmp_path, monkeypatch, f"echo $$ > {pidfile}\nsleep 30\n")
    cfg = Configuration(_env_file=None)
    cfg.workspace_root = str(tmp_path / "ws")
    cfg.home_dir = str(tmp_path / "home")
    s = ClaudeSession("ctx-1", cfg.workspace_root)
    translator = _mk_translator()

    task = asyncio.ensure_future(run_turn(s, "hello", translator, cfg))
    # Wait until the subprocess has spawned and recorded its PID.
    for _ in range(50):
        if pidfile.exists():
            break
        await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    pid = int(pidfile.read_text().strip())
    for _ in range(20):  # give the kill a moment to take effect
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
