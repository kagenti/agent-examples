"""
Module for A2A Agent.
"""

import asyncio
import logging
import os
import sys
import threading
import traceback

import uvicorn
from crewai_tools.adapters.tool_collection import ToolCollection


from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.helpers import new_task_from_user_message, new_text_part
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskState,
    SecurityScheme,
    HTTPAuthSecurityScheme,
)

from starlette.applications import Starlette

from git_issue_agent.config import settings, Settings
from git_issue_agent.event import Event
from git_issue_agent.main import GitIssueAgent, TaskCancelled
from git_issue_agent.mcp_connect import mcp_tools_session

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.LOG_LEVEL, stream=sys.stdout, format="%(levelname)s: %(message)s")


def get_agent_card(host: str, port: int):
    """Returns the Agent Card for the AG2 Agent."""
    capabilities = AgentCapabilities(streaming=True)
    skill = AgentSkill(
        id="github_issue_agent",
        name="Github issue agent",
        description="Answer queries about Github issues",
        tags=["git", "github", "issues"],
        examples=[
            "Find me the issues with the most comments in kubernetes/kubernetes",
            "Show all issues assigned to me across any repository",
        ],
    )
    # Allow env var AGENT_ENDPOINT to override the URL in the agent card
    url = os.getenv("AGENT_ENDPOINT", f"http://{host}:{port}").rstrip("/") + "/"
    return AgentCard(
        name="Github issue agent",
        description="Answer queries about Github issues",
        supported_interfaces=[AgentInterface(url=url, protocol_binding="JSONRPC")],
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
        security_schemes={
            "Bearer": SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(
                    scheme="bearer", bearer_format="JWT", description="OAuth 2.0 JWT token"
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
            parts = [new_text_part(message)]
            await self.task_updater.add_artifact(parts)
            await self.task_updater.complete()
        else:
            await self.task_updater.update_status(
                TaskState.TASK_STATE_WORKING,
                self.task_updater.new_agent_message([new_text_part(message)]),
            )


class GithubExecutor(AgentExecutor):
    """
    A class to handle research execution for A2A Agent.
    """

    def __init__(self):
        # Per-request cooperative-cancel flags, keyed by task id. CrewAI runs its
        # crew synchronously in a worker thread (kickoff_async -> asyncio.to_thread),
        # so cancelling the awaiting coroutine cannot stop the work; instead the
        # crew checks this Event between steps and stops itself.
        self._cancel_events: dict[str, threading.Event] = {}

    async def _run_agent(
        self,
        messages: dict,
        settings: Settings,
        event_emitter: Event,
        toolkit: ToolCollection,
        cancel_event: threading.Event,
    ):
        git_issue_agent = GitIssueAgent(
            config=settings,
            eventer=event_emitter,
            mcp_toolkit=toolkit,
            cancel_event=cancel_event,
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
        # If GITHUB_TOKEN is set, pass it as Bearer header to MCP.
        # If not set, assume AuthBridge handles auth transparently (envoy injects tokens).
        headers = {}
        if settings.GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {settings.GITHUB_TOKEN}"
        elif context.call_context and (context.call_context.state or {}).get("headers", {}).get("authorization"):
            headers["Authorization"] = context.call_context.state["headers"]["authorization"]
        else:
            logging.warning(
                "No GITHUB_TOKEN or inbound Authorization header; outbound requests will be unauthenticated"
            )

        user_input = [context.get_user_input()]
        task = context.current_task
        if not task:
            task = new_task_from_user_message(context.message)
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

        # Register a cooperative-cancel flag for this task so cancel()/disconnect
        # can stop the (thread-bound) crew run.
        cancel_event = threading.Event()
        self._cancel_events[task.id] = cancel_event

        # Hook up MCP tools
        try:
            if settings.MCP_URL:
                logging.info("Connecting to MCP server at %s", settings.MCP_URL)

                server_params = {
                    "url": settings.MCP_URL,
                    "transport": "streamable-http",
                    "headers": headers,
                }
                # mcp_tools_session fails fast if the connection errors out, while
                # still allowing up to MCP_TIMEOUT for a slow (e.g. OAuth) connect.
                async with mcp_tools_session(
                    server_params,
                    connect_timeout=settings.MCP_TIMEOUT,
                    poll_interval=settings.MCP_POLL_INTERVAL,
                ) as mcp_tools:
                    # Keep only search and list issue-related tools.
                    issue_tools = [
                        tool
                        for tool in mcp_tools
                        if ("issue" in tool.name.lower() or "label" in tool.name.lower())
                        and ("search" in tool.name.lower() or "list" in tool.name.lower())
                    ]

                    if not issue_tools:
                        raise RuntimeError(
                            "No issue-related tools found from the GitHub MCP server. "
                            "Ensure your PAT scopes allow issue access and the server is reachable."
                        )
                    # Tool output is bounded inside GitIssueAgent (wrap_tool_output).
                    await self._run_agent(messages, settings, event_emitter, issue_tools, cancel_event)
            else:
                await self._run_agent(messages, settings, event_emitter, None, cancel_event)

        except (asyncio.CancelledError, TaskCancelled):
            # A2A raises CancelledError in this task on cancel/disconnect; the crew
            # may also surface TaskCancelled after observing the flag. Signal the
            # worker thread to stop and report the task as cancelled.
            cancel_event.set()
            logging.info("Task %s cancelled; stopping crew run", task.id)
            try:
                await task_updater.cancel()
            except Exception:  # noqa: BLE001 - best-effort status on an already-torn-down queue
                pass
            raise
        except Exception as e:
            traceback.print_exc()
            await event_emitter.emit_event(
                f"I'm sorry I was unable to fulfill your request. I encountered the following exception: {str(e)}", True
            )
        finally:
            self._cancel_events.pop(task.id, None)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Signal the running crew for this task to stop, and report cancellation.

        The crew runs in a worker thread, so we flip its cooperative-cancel flag;
        it stops at the next step boundary rather than mid-LLM-call.
        """
        task = context.current_task
        task_id = task.id if task else None
        cancel_event = self._cancel_events.get(task_id) if task_id else None
        if cancel_event is not None:
            cancel_event.set()
            logging.info("Cancellation requested for task %s", task_id)

        if task_id:
            task_updater = TaskUpdater(event_queue, task_id, task.context_id)
            await task_updater.cancel()


def run():
    """
    Runs the A2A Agent application.
    """
    agent_card = get_agent_card(host="0.0.0.0", port=settings.SERVICE_PORT)

    request_handler = DefaultRequestHandler(
        agent_executor=GithubExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    # a2a-sdk 1.x replaced A2AStarletteApplication with route factories that we
    # assemble into a Starlette app ourselves.
    # enable_v0_3_compat is needed because Rossoctl uses A2A 0.3 client libraries
    routes = create_jsonrpc_routes(request_handler, rpc_url="/", enable_v0_3_compat=True)
    # Serve the current well-known path (/.well-known/agent-card.json) plus the
    # legacy /.well-known/agent.json path for backward compatibility.
    routes += create_agent_card_routes(agent_card)
    routes += create_agent_card_routes(agent_card, card_url="/.well-known/agent.json")

    app = Starlette(routes=routes)

    uvicorn.run(app, host="0.0.0.0", port=settings.SERVICE_PORT)
