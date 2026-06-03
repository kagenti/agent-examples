import asyncio
import os
import shutil
import time
import uuid
from collections import OrderedDict

# Stable namespace so a given context_id always derives the same session UUID,
# even across process restarts. Generated once; do not change.
_NAMESPACE = uuid.UUID("6f8d2c4e-1b3a-4f5e-9a7b-0c1d2e3f4a5b")


class ClaudeSession:
    """One Claude Code conversation, isolated by its own working directory."""

    def __init__(self, context_id: str, workspace_root: str):
        self.context_id = context_id
        # `claude --session-id` requires a valid UUID; uuid5 guarantees one.
        self.session_uuid = str(uuid.uuid5(_NAMESPACE, context_id))
        self.workdir = os.path.join(workspace_root, self.session_uuid)
        self.started = False  # False → use --session-id; True → use --resume
        self.lock = asyncio.Lock()  # serialize turns within this session
        self.last_used = 0.0

    def ensure_workdir(self) -> str:
        os.makedirs(self.workdir, exist_ok=True)
        return self.workdir

    def cleanup(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)


class SessionRegistry:
    """Maps A2A context_id → ClaudeSession with an LRU cap on live sessions."""

    def __init__(self, workspace_root: str, max_sessions: int = 100):
        self._workspace_root = workspace_root
        self._max_sessions = max_sessions
        self._sessions: "OrderedDict[str, ClaudeSession]" = OrderedDict()
        self._guard = asyncio.Lock()

    async def get_or_create(self, context_id: str) -> ClaudeSession:
        async with self._guard:
            session = self._sessions.get(context_id)
            if session is None:
                session = ClaudeSession(context_id, self._workspace_root)
                self._sessions[context_id] = session
                self._evict_if_needed(protected=context_id)
            self._sessions.move_to_end(context_id)
            session.last_used = time.monotonic()
            return session

    async def context_ids(self) -> list[str]:
        async with self._guard:
            return list(self._sessions.keys())

    def _evict_if_needed(self, protected: str | None = None) -> None:
        # Evict least-recently-used sessions that are not mid-turn. Never evict
        # the just-created session (`protected`); if every other session is
        # busy, allow temporary overflow rather than corrupt one.
        while len(self._sessions) > self._max_sessions:
            victim_key = None
            for key, sess in self._sessions.items():  # oldest first
                if key == protected:
                    continue
                if not sess.lock.locked():
                    victim_key = key
                    break
            if victim_key is None:
                break
            victim = self._sessions.pop(victim_key)
            victim.cleanup()
