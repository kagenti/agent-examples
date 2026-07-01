import json
import logging

from a2a.helpers import new_text_part
from a2a.types import TaskState

logger = logging.getLogger(__name__)

_TRUNC = 256


def _truncate(value, n: int = _TRUNC) -> str:
    s = str(value)
    return s if len(s) <= n else s[:n] + "..."


class StreamTranslator:
    """Translates Claude `stream-json` events into A2A task updates.

    Intermediate events (assistant text, tool calls/results) become `working`
    status updates. The terminal `result` event is accumulated and emitted by
    `finish()` as either a completed artifact or a failed task.
    """

    def __init__(self, task_updater):
        self._tu = task_updater
        self.session_id: str | None = None
        self.final_text: str | None = None
        self.errored: bool = False
        self.error_reason: str | None = None

    async def _working(self, text: str) -> None:
        if not text.strip():
            return
        await self._tu.update_status(
            TaskState.TASK_STATE_WORKING,
            self._tu.new_agent_message([new_text_part(text)]),
        )

    async def handle(self, event: dict) -> None:
        etype = event.get("type")
        if etype == "system" and event.get("subtype") == "init":
            # Capture the session id, but don't emit a status update: every turn
            # (including each --resume) emits an init event, so surfacing it would
            # repeat "session started" on every message.
            self.session_id = event.get("session_id")
        elif etype == "assistant":
            for block in event.get("message", {}).get("content", []):
                btype = block.get("type")
                if btype == "text":
                    await self._working(block.get("text", ""))
                elif btype == "tool_use":
                    name = block.get("name", "tool")
                    await self._working(f"🔧 {name}: {_truncate(json.dumps(block.get('input', {})))}")
        elif etype == "user":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
                    await self._working(f"✓ {_truncate(content)}")
        elif etype == "result":
            if event.get("subtype") == "success" and not event.get("is_error"):
                self.final_text = event.get("result", "")
            else:
                self.errored = True
                self.error_reason = event.get("subtype") or "unknown error"

    async def finish(self) -> None:
        """Emit the terminal A2A state based on what was accumulated."""
        if self.errored or self.final_text is None:
            reason = self.error_reason or "Claude produced no result"
            await self._tu.add_artifact([new_text_part(f"Error: {reason}")])
            await self._tu.failed()
        else:
            await self._tu.add_artifact([new_text_part(self.final_text)])
            await self._tu.complete()
