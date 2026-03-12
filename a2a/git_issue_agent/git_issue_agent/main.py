from crewai_tools.adapters.tool_collection import ToolCollection

import logging
import sys

from git_issue_agent.config import Settings, settings
from git_issue_agent.event import Event
from git_issue_agent.agents import GitAgents

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.LOG_LEVEL, stream=sys.stdout, format="%(levelname)s: %(message)s")


class GitIssueAgent:
    def __init__(
        self,
        config: Settings,
        eventer: Event = None,
        mcp_toolkit: ToolCollection = None,
        logger=None,
    ):
        self.agents = GitAgents(settings, mcp_toolkit)
        self.eventer = eventer
        self.logger = logger or logging.getLogger(__name__)

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

    async def execute(self, user_input):
        query = self.extract_user_input(user_input)
        await self._send_event("🧐 Evaluating requirements...")
        await self.agents.prereq_id_crew.kickoff_async(inputs={"request": query, "repo": "", "owner": "", "issues": []})
        repo_id_task_output = self.agents.prereq_identifier_task.output.pydantic

        if repo_id_task_output.issue_numbers:
            if not repo_id_task_output.owner or not repo_id_task_output.repo:
                return "When supplying issue numbers, you must provide both a repository name and owner."
        if repo_id_task_output.repo:
            if not repo_id_task_output.owner:
                return "When supplying a repository name, you must also provide an owner of the repo."

        await self._send_event("🔎 Searching for issues...")
        await self.agents.crew.kickoff_async(
            inputs={
                "request": query,
                "owner": repo_id_task_output.owner,
                "repo": repo_id_task_output.repo,
                "issues": repo_id_task_output.issue_numbers,
            }
        )
        return self.agents.issue_query_task.output.raw
