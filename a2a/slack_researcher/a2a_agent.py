"""
Module for A2A Agent.
"""

import logging
import sys
import traceback
from typing import Callable

import uvicorn
from autogen.mcp.mcp_client import create_toolkit, Toolkit
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TaskState, TextPart, SecurityScheme, HTTPAuthSecurityScheme
from a2a.utils import new_agent_text_message, new_task

from starlette.middleware.authentication import AuthenticationMiddleware

from slack_researcher.config import settings, Settings
from slack_researcher.event import Event
from slack_researcher.main import SlackAgent
from slack_researcher.auth import on_auth_error, BearerAuthBackend, auth_headers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s: %(message)s')

def get_agent_card(host: str, port: int):
    """Returns the Agent Card for the AG2 Agent."""
    capabilities = AgentCapabilities(streaming=True)
    skill = AgentSkill(
        id="slack_researcher",
        name="Slack research agent",
        description="Answer queries by searching through a given slack server",
        tags=["research", "slack", "search", "report"],
        examples=[
            "Find me the most popular channels for discussing AI agents",
            "Summarize what's been happening in the general channel lately",
        ],
    )
    return AgentCard(
        name="Web Research Agent",
        description="Answer queries by searching through a given slack server",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
        securitySchemes={
            "Bearer": SecurityScheme(
                root=HTTPAuthSecurityScheme(
                    type="http",
                    scheme="bearer",
                    bearerFormat="JWT",
                    description="OAuth 2.0 JWT token"
                )
            )
        },
    )


class A2AEvent(Event):
    """
    A class to handle events for A2A Agent.

    Attributes:
        task_updater (TaskUpdater): The task updater instance.
    """

    def __init__(self, task_updater: TaskUpdater):
        """
        Initializes the A2AEvent instance.

        Args:
            task_updater (TaskUpdater): The task updater instance.
        """
        self.task_updater = task_updater

    async def emit_event(self, message: str, final: bool = False) -> None:
        """
        Emits an event with the given message.

        Args:
            message (str): The event message.
            final (bool): Whether the event is final. Defaults to False.
        """
        logger.info("Emitting event %s", message)

        if final:
            parts = [TextPart(text=message)]
            await self.task_updater.add_artifact(parts)
            await self.task_updater.complete()
        else:
            await self.task_updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    message,
                    self.task_updater.context_id,
                    self.task_updater.task_id,
                ),
            )


class ResearchExecutor(AgentExecutor):
    """
    A class to handle research execution for A2A Agent.
    """
    async def _run_agent(self,
        messages: dict,
        settings: Settings,
        event_emitter: Event,
        assistant_tool_map: dict[str, Callable],
        toolkit: Toolkit):

        slack_agent = SlackAgent(
            config=settings,
            eventer=event_emitter,
            assistant_tools=assistant_tool_map,
            mcp_toolkit=toolkit,
        )
        result = await slack_agent.execute(messages)
        await event_emitter.emit_event(result, True)

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        Executes the research task.

        Args:
            context (RequestContext): The request context.
            event_queue (EventQueue): The event queue instance.

        Returns:
            None
        """
        user_token = context.call_context.user._user.access_token
        user_input = [context.get_user_input()]
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)
        event_emitter = A2AEvent(task_updater)
        messages = []
        for message in user_input:
            messages.append(
                {
                    "role": "User",
                    "content": message,
                }
            )

        # no internal tools right now, will add later
        assistant_tool_map = {}

        # Hook up MCP tools
        toolkit = None
        try:
            if settings.MCP_URL:
                logging.info("Connecting to MCP server at %s", settings.MCP_URL)

                headers = await auth_headers(
                    user_token, 
                    target_scopes=settings.TARGET_SCOPES
                )

                async with streamablehttp_client(
                    url=settings.MCP_URL,
                    headers=headers
                )  as (
                    read_stream,
                    write_stream,
                    _,
                ), ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    toolkit = await create_toolkit(
                        session=session, use_mcp_resources=False
                    )
                    await self._run_agent(messages, settings,
                        event_emitter,
                        assistant_tool_map,
                        toolkit,)
            else:
                await self._run_agent(messages, settings,
                    event_emitter,
                    assistant_tool_map,
                    toolkit,)

        except Exception as e:
            traceback.print_exc()
            await event_emitter.emit_event(f"I'm sorry I was unable to fulfill your request. I encountered the following exception: {str(e)}", True)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Not implemented
        """
        raise Exception("cancel not supported")


def run():
    """
    Runs the A2A Agent application.
    """
    agent_card = get_agent_card(host="0.0.0.0", port=settings.SERVICE_PORT)

    request_handler = DefaultRequestHandler(
        agent_executor=ResearchExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    app = server.build()  # this returns a Starlette app
    # if one of the auth variables is set, create middleware
    # if none of them are set, ignore all authorization headers. No token validation will be performed
    if not settings.JWKS_URI is None:
        logging.info("JWKS_URI is set - using JWT Validation middleware")
        app.add_middleware(AuthenticationMiddleware, backend=BearerAuthBackend(), on_error=on_auth_error)

    uvicorn.run(app, host="0.0.0.0", port=settings.SERVICE_PORT)
