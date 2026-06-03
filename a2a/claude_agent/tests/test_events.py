from unittest.mock import AsyncMock, MagicMock

from claude_agent.events import StreamTranslator


def make_translator():
    tu = MagicMock()
    tu.context_id = "ctx-1"
    tu.task_id = "task-1"
    tu.update_status = AsyncMock()
    tu.add_artifact = AsyncMock()
    tu.complete = AsyncMock()
    tu.failed = AsyncMock()
    return StreamTranslator(tu), tu


async def test_init_event_captures_session_id():
    t, tu = make_translator()
    await t.handle({"type": "system", "subtype": "init", "session_id": "abc"})
    assert t.session_id == "abc"
    tu.update_status.assert_awaited()  # emitted a "started" working update


async def test_assistant_text_emits_working_update():
    t, tu = make_translator()
    await t.handle(
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Hello there"}]},
        }
    )
    tu.update_status.assert_awaited()


async def test_tool_use_emits_working_update():
    t, tu = make_translator()
    await t.handle(
        {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}]},
        }
    )
    tu.update_status.assert_awaited()


async def test_success_result_completes_with_artifact():
    t, tu = make_translator()
    await t.handle(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "final answer",
            "session_id": "abc",
        }
    )
    await t.finish()
    assert t.final_text == "final answer"
    tu.add_artifact.assert_awaited()
    tu.complete.assert_awaited()
    tu.failed.assert_not_awaited()


async def test_error_result_fails():
    t, tu = make_translator()
    await t.handle(
        {
            "type": "result",
            "subtype": "error_max_turns",
            "is_error": True,
        }
    )
    await t.finish()
    assert t.errored is True
    tu.failed.assert_awaited()
    tu.complete.assert_not_awaited()


async def test_no_result_is_treated_as_failure():
    t, tu = make_translator()
    await t.finish()  # nothing handled
    tu.failed.assert_awaited()
