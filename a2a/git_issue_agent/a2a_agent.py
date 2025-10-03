"""
Module for A2A Agent.
"""

import logging
import sys
import traceback
from typing import Callable

import uvicorn
from crewai_tools import MCPServerAdapter
from crewai_tools.adapters.tool_collection import ToolCollection

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TaskState, TextPart, SecurityScheme, HTTPAuthSecurityScheme
from a2a.utils import new_agent_text_message, new_task

from starlette.authentication import AuthCredentials, SimpleUser, AuthenticationBackend
from starlette.middleware.authentication import AuthenticationMiddleware

from git_issue_agent.auth import on_auth_error, BearerAuthBackend, auth_headers
from git_issue_agent.config import settings, Settings
from git_issue_agent.event import Event
from git_issue_agent.main import GitIssueAgent

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.LOG_LEVEL, stream=sys.stdout, format='%(levelname)s: %(message)s')

class BearerAuthBackend(AuthenticationBackend):
    """ Very temporary demo to grab auth token and print it"""
    async def authenticate(self, conn):
        try:
            auth = conn.headers.get("authorization")
            if not auth or not auth.lower().startswith("bearer "):
                print("No bearer token provided")
                return
            token = auth.split(" ", 1)[1]
            print(f"TOKEN: {token}")

            # Storing the token as the username - not a real life scenario - just demo-ing the passing of creds
            user = SimpleUser(token)
            return AuthCredentials(["authenticated"]), user
        except Exception as e:
            logger.error("Exception when attempting to obtain user token")
            logger.error(e)
    

def get_agent_card(host: str, port: int):
    """Returns the Agent Card for the AG2 Agent."""
    capabilities = AgentCapabilities(streaming=True)
    skill = AgentSkill(
        id="github_issue_agent",
        name="Github issue agent",
        description="Answer queries by searching through a given slack server",
        tags=["git", "github", "issues"],
        examples=[
            "Find me the issues with the most comments in kubernetes/kubernetes",
            "Show all issues assigned to me across any repository",
        ],
    )
    return AgentCard(
        name="Github issue agent",
        description="Answer queries about Github issues",
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


class GithubExecutor(AgentExecutor):
    """
    A class to handle research execution for A2A Agent.
    """
    async def _run_agent(self,
        messages: dict,
        settings: Settings,
        event_emitter: Event,
        toolkit: ToolCollection):

        git_issue_agent = GitIssueAgent(
            config=settings,
            eventer=event_emitter,
            mcp_toolkit=toolkit,
        )
        result = await git_issue_agent.execute(messages)
        await event_emitter.emit_event(result, True)

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        Executes the task.

        Args:
            context (RequestContext): The request context.
            event_queue (EventQueue): The event queue instance.

        Returns:
            None
        """
        if settings.JWKS_URI:
            user_token = context.call_context.user._user.access_token
        else:
            user_token = settings.GITHUB_TOKEN
        user_input = [context.get_user_input()]
        task = context.current_task
        if not task:
            task = new_task(context.message)
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

        # Hook up MCP tools
        try:
            if settings.MCP_URL:
                logging.info("Connecting to MCP server at %s", settings.MCP_URL)

                headers = await auth_headers(
                    user_token, 
                    target_audience=settings.TARGET_AUDIENCE, 
                    target_scopes=settings.TARGET_SCOPES
                )

                server_params = {
                    "url": settings.MCP_URL,
                    "transport": "streamable-http",
                    "headers": headers,
                }
                with MCPServerAdapter(server_params, connect_timeout=60) as mcp_tools:
                    # Keep only search and list issue-related tools.
                    issue_tools = [
                        tool
                        for tool in mcp_tools
                        if "issue" in tool.name.lower() and ("search" in tool.name.lower() or "list" in tool.name.lower())
                    ]

                    if not issue_tools:
                        raise RuntimeError(
                            "No issue-related tools found from the GitHub MCP server. "
                            "Ensure your PAT scopes allow issue access and the server is reachable."
                        )
                    await self._run_agent(messages, settings, event_emitter, issue_tools)
            else:
                await self._run_agent(messages, settings,
                    event_emitter,
                    None)

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
        agent_executor=GithubExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    app = server.build()  # this returns a Starlette app
    if not settings.JWKS_URI is None:
        logging.info("JWKS_URI is set - using JWT Validation middleware")
    app.add_middleware(AuthenticationMiddleware, backend=BearerAuthBackend(), on_error=on_auth_error)

    uvicorn.run(app, host="0.0.0.0", port=settings.SERVICE_PORT)
