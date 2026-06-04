import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent.agent import ClaudeAgentExecutor, get_agent_card
from claude_agent.configuration import Configuration
from claude_agent.session import SessionRegistry


def test_agent_card_has_streaming_and_url():
    card = get_agent_card("0.0.0.0", 8000)
    assert card.capabilities.streaming is True
    assert card.url.endswith("/")
    assert card.skills  # at least one skill advertised


async def test_executor_runs_turn_under_lock_and_finishes(tmp_path):
    cfg = Configuration(_env_file=None)
    cfg.workspace_root = str(tmp_path / "ws")
    registry = SessionRegistry(cfg.workspace_root, max_sessions=10)
    semaphore = asyncio.Semaphore(2)
    executor = ClaudeAgentExecutor(cfg, registry, semaphore)

    # Build a fake RequestContext + EventQueue.
    context = MagicMock()
    context.current_task = MagicMock(id="task-1", context_id="ctx-1")
    context.get_user_input.return_value = "do the thing"
    event_queue = MagicMock()
    event_queue.enqueue_event = AsyncMock()

    # Patch run_turn so we assert it's called and simulate a successful result.
    async def fake_run_turn(session, prompt, translator, config):
        translator.final_text = "ok"
        session.started = True

    with (
        patch("claude_agent.agent.run_turn", side_effect=fake_run_turn) as rt,
        patch("claude_agent.agent.StreamTranslator") as ST,
    ):
        translator = MagicMock()
        translator.finish = AsyncMock()
        ST.return_value = translator
        await executor.execute(context, event_queue)

    rt.assert_awaited_once()
    translator.finish.assert_awaited_once()


async def test_cancel_raises_not_implemented(tmp_path):
    cfg = Configuration(_env_file=None)
    cfg.workspace_root = str(tmp_path / "ws")
    executor = ClaudeAgentExecutor(cfg, SessionRegistry(cfg.workspace_root), asyncio.Semaphore(1))
    with pytest.raises(NotImplementedError):
        await executor.cancel(MagicMock(), MagicMock())
