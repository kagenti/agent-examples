import os

from claude_code_agent.session import ClaudeSession, SessionRegistry


def test_session_uuid_is_deterministic_and_valid():
    import uuid

    a = ClaudeSession("ctx-abc", "/workspace")
    b = ClaudeSession("ctx-abc", "/workspace")
    c = ClaudeSession("ctx-different", "/workspace")
    assert a.session_uuid == b.session_uuid
    assert a.session_uuid != c.session_uuid
    # Must be a valid UUID string (required by `claude --session-id`).
    uuid.UUID(a.session_uuid)


def test_workdir_is_per_session(tmp_path):
    a = ClaudeSession("ctx-1", str(tmp_path))
    b = ClaudeSession("ctx-2", str(tmp_path))
    assert a.workdir != b.workdir
    a.ensure_workdir()
    assert os.path.isdir(a.workdir)


async def test_get_or_create_returns_same_object(tmp_path):
    reg = SessionRegistry(str(tmp_path), max_sessions=10)
    s1 = await reg.get_or_create("ctx-1")
    s2 = await reg.get_or_create("ctx-1")
    assert s1 is s2


async def test_lru_eviction_removes_workdir(tmp_path):
    reg = SessionRegistry(str(tmp_path), max_sessions=2)
    s1 = await reg.get_or_create("ctx-1")
    s1.ensure_workdir()
    await reg.get_or_create("ctx-2")
    await reg.get_or_create("ctx-3")  # exceeds cap → evict LRU (ctx-1)
    assert not os.path.isdir(s1.workdir)
    # A fresh ctx-1 is a new object after eviction.
    s1b = await reg.get_or_create("ctx-1")
    assert s1b is not s1


async def test_eviction_skips_locked_sessions(tmp_path):
    reg = SessionRegistry(str(tmp_path), max_sessions=1)
    busy = await reg.get_or_create("busy")
    async with busy.lock:
        # 'busy' is in use; adding another must not evict it.
        other = await reg.get_or_create("other")
        ids = await reg.context_ids()
    assert "busy" in ids
    assert other.context_id == "other"
