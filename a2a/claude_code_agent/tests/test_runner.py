from unittest.mock import AsyncMock, MagicMock

import pytest

from claude_code_agent.configuration import Configuration
from claude_code_agent.events import StreamTranslator
from claude_code_agent.runner import build_argv, build_env, run_turn
from claude_code_agent.session import ClaudeSession


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


def test_build_env_sets_litellm_wiring():
    cfg = Configuration(_env_file=None)
    cfg.anthropic_auth_token = "sk-x"
    env = build_env(cfg)
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-x"
    assert env["ANTHROPIC_BASE_URL"] == cfg.anthropic_base_url
    assert env["ANTHROPIC_MODEL"] == "sonnet"
    assert env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] == "1"
    assert env["HOME"] == cfg.home_dir


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
