import logging
import os
import re
from textwrap import dedent

import uvicorn
from langchain_core.messages import HumanMessage
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Route

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TaskState, TextPart
from a2a.utils import new_agent_text_message, new_task
from weather_service.configuration import Configuration
from weather_service.graph import get_graph, get_mcpclient
from weather_service.observability import (
    create_tracing_middleware,
    get_root_span,
    set_span_output,
)


class SecretRedactionFilter(logging.Filter):
    """Redacts Bearer tokens and the configured API key from log messages."""

    _BEARER_RE = re.compile(r"(Bearer\s+)\S+", re.IGNORECASE)

    def __init__(self):
        super().__init__()
        key = os.environ.get("LLM_API_KEY", "").strip()
        self._key_re = re.compile(re.escape(key)) if len(key) > 8 else None

    def _redact(self, text: str) -> str:
        text = self._BEARER_RE.sub(r"\1[REDACTED]", text)
        if self._key_re:
            text = self._key_re.sub("[REDACTED]", text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._redact(record.msg)
        if isinstance(record.args, dict):
            record.args = {k: self._redact(v) if isinstance(v, str) else v for k, v in record.args.items()}
        elif isinstance(record.args, tuple):
            record.args = tuple(self._redact(a) if isinstance(a, str) else a for a in record.args)
        return True


logging.basicConfig(level=logging.INFO)
logging.getLogger().addFilter(SecretRedactionFilter())
logger = logging.getLogger(__name__)


def get_agent_card(host: str, port: int):
    """Returns the Agent Card for the AG2 Agent."""
    capabilities = AgentCapabilities(streaming=True)
    skill = AgentSkill(
        id="weather_assistant",
        name="Weather Assistant",
        description="**Weather Assistant** – Personalized assistant for weather info.",
        tags=["weather"],
        examples=[
            "What is the weather in NY?",
            "What is the weather in Rome?",
        ],
    )
    return AgentCard(
        name="Weather Assistant",
        description=dedent(
            """\
            This agent provides a simple weather information assistance.

            ## Input Parameters
            - **prompt** (string) – the city for which you want to know weather info.

            ## Key Features
            - **MCP Tool Calling** – uses a MCP tool to get weather info.
            """,
        ),
        # Allow env var AGENT_ENDPOINT to override the URL in the agent card
        url=os.getenv("AGENT_ENDPOINT", f"http://{host}:{port}").rstrip("/") + "/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
    )


class A2AEvent:
    """
    A class to handle events for A2A Agent.

    Attributes:
        task_updater (TaskUpdater): The task updater instance.
    """

    def __init__(self, task_updater: TaskUpdater):
        self.task_updater = task_updater

    async def emit_event(self, message: str, final: bool = False, failed: bool = False) -> None:
        logger.info("Emitting event %s", message)

        if final or failed:
            parts = [TextPart(text=message)]
            await self.task_updater.add_artifact(parts)
            if final:
                await self.task_updater.complete()
            if failed:
                await self.task_updater.failed()
        else:
            await self.task_updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    message,
                    self.task_updater.context_id,
                    self.task_updater.task_id,
                ),
            )


class WeatherExecutor(AgentExecutor):
    """
    A class to handle weather assistant execution for A2A Agent.
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        The agent allows to retrieve weather info through a natural language conversational interface
        """

        # Setup Event Emitter
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)
        event_emitter = A2AEvent(task_updater)

        # Check API key before attempting LLM calls
        config = Configuration()
        if not config.has_valid_api_key:
            await event_emitter.emit_event(
                "Error: No LLM API key configured. Set the LLM_API_KEY environment variable.",
                failed=True,
            )
            return

        # Get user input for the agent
        user_input = context.get_user_input()

        # Parse Messages
        messages = [HumanMessage(content=user_input)]
        input = {"messages": messages}
        logger.info(f"Processing messages: {input}")

        # Note: Root span with MLflow attributes is created by tracing middleware
        # Here we just run the agent logic - spans from LangChain are auto-captured
        output = None

        # Forward inbound Authorization header to outbound MCP tool calls.
        # This enables transparent token exchange when deployed behind a waypoint
        # or AuthBridge proxy (same pattern as git_issue_agent, see c8ebde1).
        mcp_headers = None
        if context.call_context and (context.call_context.state or {}).get("headers", {}).get("authorization"):
            mcp_headers = {"Authorization": context.call_context.state["headers"]["authorization"]}
            logger.info("Forwarding inbound Authorization header to MCP tool calls")
        else:
            logger.warning("No inbound Authorization header; MCP tool calls will be unauthenticated")

        # Test MCP connection first
        logger.info(f"Attempting to connect to MCP server at: {os.getenv('MCP_URL', 'http://localhost:8000/sse')}")

        mcpclient = get_mcpclient(headers=mcp_headers)

        # Try to get tools to verify connection
        try:
            tools = await mcpclient.get_tools()
            logger.info(f"Successfully connected to MCP server. Available tools: {[tool.name for tool in tools]}")
        except Exception as tool_error:
            logger.error(f"Failed to connect to MCP server: {tool_error}")
            await event_emitter.emit_event(
                f"Error: Cannot connect to MCP weather service at {os.getenv('MCP_URL', 'http://localhost:8000/sse')}. Please ensure the weather MCP server is running. Error: {tool_error}",
                failed=True,
            )
            return

        try:
            graph = await get_graph(mcpclient)
        except Exception as graph_error:
            logger.error(f"Failed to create LLM graph: {graph_error}")
            await event_emitter.emit_event(f"Error: Failed to initialize LLM graph: {graph_error}", failed=True)
            return

        try:
            async for event in graph.astream(input, stream_mode="updates"):
                await event_emitter.emit_event(
                    "\n".join(
                        f"🚶‍♂️{key}: {str(value)[:256] + '...' if len(str(value)) > 256 else str(value)}"
                        for key, value in event.items()
                    )
                    + "\n"
                )
                output = event
                logger.info(f"event: {event}")
        except Exception as llm_error:
            logger.error(f"LLM execution failed: {llm_error}")
            await event_emitter.emit_event(f"Error: LLM execution failed: {llm_error}", failed=True)
            return

        output = output.get("assistant", {}).get("final_answer") if output else None

        # Set span output BEFORE emitting final event (for streaming response capture)
        # This populates mlflow.spanOutputs, output.value, gen_ai.completion
        # Use get_root_span() to get the middleware-created root span, not the
        # current A2A span (trace.get_current_span() would return wrong span)
        if output:
            root_span = get_root_span()
            if root_span and root_span.is_recording():
                set_span_output(root_span, str(output))

        await event_emitter.emit_event(str(output), final=True)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Not implemented
        """
        raise Exception("cancel not supported")


def run():
    """
    Runs the A2A Agent application.
    """
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    agent_card = get_agent_card(host=host, port=port)

    request_handler = DefaultRequestHandler(
        agent_executor=WeatherExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # Build the Starlette app
    app = server.build()

    # Add the new agent-card.json path alongside the legacy agent.json path
    app.routes.insert(
        0,
        Route(
            "/.well-known/agent-card.json",
            server._handle_get_agent_card,
            methods=["GET"],
            name="agent_card_new",
        ),
    )

    # Add tracing middleware - creates root span with MLflow/GenAI attributes
    app.add_middleware(BaseHTTPMiddleware, dispatch=create_tracing_middleware())

    uvicorn.run(app, host=host, port=port)
