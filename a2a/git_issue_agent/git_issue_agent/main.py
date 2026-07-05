import json
import logging
import re
import sys
import threading

from crewai_tools.adapters.tool_collection import ToolCollection

from git_issue_agent.agents import GitAgents
from git_issue_agent.config import Settings, settings
from git_issue_agent.data_types import IssueSearchInfo
from git_issue_agent.event import Event
from git_issue_agent.tool_limits import wrap_tool_output

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.LOG_LEVEL, stream=sys.stdout, format="%(levelname)s: %(message)s")


class TaskCancelled(Exception):
    """Raised to unwind a CrewAI run once cancellation has been requested."""


def _parse_prereq_from_raw(raw: str) -> IssueSearchInfo:
    """Parse IssueSearchInfo from raw LLM text when instructor/pydantic parsing fails.

    Some Ollama models don't produce structured tool calls that crewai's instructor
    integration expects. This fallback extracts JSON from the raw text output.
    """
    # Only matches flat JSON (no nested braces). Sufficient for the current
    # IssueSearchInfo schema; revisit if the model gains nested fields.
    json_match = re.search(r"\{[^{}]*\}", raw)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return IssueSearchInfo(**data)
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning("Could not parse prereq JSON from raw output: %s", raw)
    return IssueSearchInfo()


class GitIssueAgent:
    def __init__(
        self,
        config: Settings,
        eventer: Event = None,
        mcp_toolkit: ToolCollection = None,
        logger=None,
        cancel_event: threading.Event = None,
    ):
        self.cancel_event = cancel_event or threading.Event()
        self._truncation_notes: list[str] = []
        # Bound each MCP tool's output so a large result set can't overflow the
        # model context window (which triggers CrewAI's slow summarize fallback).
        if mcp_toolkit:
            mcp_toolkit = [
                wrap_tool_output(
                    tool,
                    max_items=config.MAX_ISSUES,
                    max_chars=config.MAX_TOOL_CHARS,
                    on_truncate=self.add_truncation_note,
                )
                for tool in mcp_toolkit
            ]
        # step_callback runs on the CrewAI worker thread after each agent step;
        # raising here propagates out of the crew loop and stops it promptly.
        self.agents = GitAgents(config, mcp_toolkit, step_callback=self._on_step)
        self.eventer = eventer
        self.logger = logger or logging.getLogger(__name__)

    def _on_step(self, _step) -> None:
        """CrewAI step callback: abort the run if cancellation was requested."""
        if self.cancel_event.is_set():
            raise TaskCancelled()

    def _check_cancelled(self) -> None:
        """Abort between crews (e.g. prereq -> main) if cancellation was requested."""
        if self.cancel_event.is_set():
            raise TaskCancelled()

    async def _send_event(self, message: str, final: bool = False):
        logger.info(message)
        if self.eventer:
            await self.eventer.emit_event(message, final)
        else:
            logger.warning("No event handler registered")

    def extract_user_input(self, body):
        content = body[-1]["content"]
        latest_content = ""

        if isinstance(content, str):
            latest_content = content
        else:
            for item in content:
                if item["type"] == "text":
                    latest_content += item["text"]
                else:
                    self.logger.warning(f"Ignoring content with type {item['type']}")

        return latest_content

    async def _get_prereq_output(self, query: str) -> IssueSearchInfo:
        """Run the prereq crew and extract IssueSearchInfo from raw text output.

        We avoid using crewai's output_pydantic because it relies on instructor's
        tool-call-based parsing, which fails with Ollama models that don't produce
        structured tool calls. Instead, we ask the LLM for JSON and parse it ourselves.
        """
        try:
            await self.agents.prereq_id_crew.kickoff_async(
                inputs={"request": query, "repo": "", "owner": "", "issues": []}
            )
            raw = self.agents.prereq_identifier_task.output.raw
            self.logger.info(f"Prereq raw output: {raw}")
            return _parse_prereq_from_raw(raw)
        except TaskCancelled:
            # Never swallow cancellation as a "prereq failed" fallback.
            raise
        except Exception as e:
            self.logger.warning(f"Prereq crew failed: {e}")
            return IssueSearchInfo()

    def add_truncation_note(self, tool_name: str, kept: int, total: int) -> None:
        """Record that a tool result was trimmed, for a user-facing note.

        Used as the ``on_truncate`` callback for wrap_tool_output. ``kept == -1``
        marks a raw character-budget cut where item counts are unknown.
        """
        if kept == -1:
            note = "Some tool output was truncated to fit the model context window."
        else:
            note = f"Results were limited to the first {kept} of {total} items."
        if note not in self._truncation_notes:
            self._truncation_notes.append(note)

    async def execute(self, user_input):
        query = self.extract_user_input(user_input)
        await self._send_event("🧐 Evaluating requirements...")
        repo_id_task_output = await self._get_prereq_output(query)

        if repo_id_task_output.issue_numbers:
            if not repo_id_task_output.owner or not repo_id_task_output.repo:
                return "When supplying issue numbers, you must provide both a repository name and owner."
        if repo_id_task_output.repo:
            if not repo_id_task_output.owner:
                return "When supplying a repository name, you must also provide an owner of the repo."

        self._check_cancelled()
        await self._send_event("🔎 Searching for issues...")
        await self.agents.crew.kickoff_async(
            inputs={
                "request": query,
                "owner": repo_id_task_output.owner,
                "repo": repo_id_task_output.repo,
                "issues": repo_id_task_output.issue_numbers,
            }
        )
        answer = self.agents.issue_query_task.output.raw
        if self._truncation_notes:
            answer = f"{answer}\n\n_Note: {' '.join(self._truncation_notes)}_"
        return answer
