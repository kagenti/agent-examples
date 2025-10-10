from dataclasses import dataclass
from crewai_tools import MCPServerAdapter
from crewai_tools.adapters.tool_collection import ToolCollection
from typing import Callable

import logging
import sys

from git_issue_agent.config import Settings, settings
from git_issue_agent.event import Event
from git_issue_agent.agents import GitAgents

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.LOG_LEVEL, stream=sys.stdout, format='%(levelname)s: %(message)s')


class GitIssueAgent:
    def __init__(self, config: Settings,
        eventer: Event = None,
        mcp_toolkit: ToolCollection = None,
        logger=None,):

        self.agents = GitAgents(settings, mcp_toolkit)
        self.eventer = eventer

    async def _send_event(self, message: str, final: bool = False):
        logger.info(message)
        if self.eventer:
            await self.eventer.emit_event(message, final)
        else:
            logger.warning("No event handler registered")

    async def execute(self, query):

        await self._send_event("🧐 Evaluating requirements...")
        await self.agents.prereq_id_crew.kickoff_async(inputs={"request": query})
        repo_id_task_output = self.agents.prereq_identifier_task.output.pydantic
        
        if not repo_id_task_output.is_owner_and_repo_identified:
            return repo_id_task_output.explanation

        await self._send_event("🔎 Searching for issues...")
        await self.agents.crew.kickoff_async(inputs={"request": query})
        return self.agents.issue_query_task.output.raw

    
        
        
