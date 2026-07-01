import logging
import os
from textwrap import dedent

import uvicorn
from langchain_core.messages import HumanMessage
from openinference.instrumentation.langchain import LangChainInstrumentor
from starlette.applications import Starlette

from a2a.helpers import new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, TaskState
from generic_agent.config import Configuration
from generic_agent.graph import get_graph, get_mcp_server_names, get_mcpclient, get_skill_folder_paths

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

LangChainInstrumentor().instrument()
config = Configuration()


def get_agent_card(host: str, port: int) -> AgentCard:
    """Returns the Agent Card for the A2A Agent."""
    try:
        mcp_names = get_mcp_server_names()
    except Exception as e:
        logger.warning(f"Failed to get MCP server names: {e}")
        mcp_names = []

    try:
        skill_folder_paths = get_skill_folder_paths()
        # Extract skill names from folder paths (last component of path)
        skill_names = [path.rstrip("/").split("/")[-1] for path in skill_folder_paths if path]
    except Exception as e:
        logger.warning(f"Failed to get skill folder paths: {e}")
        skill_names = []

    mcp_section = ""
    if mcp_names:
        mcp_section = "\n\nConnected MCP Servers:\n" + "\n".join(f"- {name}" for name in mcp_names)

    skills_section = ""
    if skill_names:
        skills_section = "\n\nAvailable Skills:\n" + "\n".join(f"- {name}" for name in skill_names)

    capabilities = AgentCapabilities(streaming=True)
    skill = AgentSkill(
        id="generic_agent",
        name="Generic Agent",
        description="**Generic Assistant** – Multi-purpose assistant for different tasks based on different MCP tools.",
        tags=mcp_names + skill_names,
        examples=[],
    )
    return AgentCard(
        name="Generic Agent",
        description=dedent(
            f"""\
            This agent provides assistance for various tasks using different MCP tools.{mcp_section}{skills_section}
            """,
        ),
        # Allow env var AGENT_ENDPOINT to override the URL in the agent card
        supported_interfaces=[
            AgentInterface(
                url=os.getenv("AGENT_ENDPOINT", f"http://{host}:{port}").rstrip("/") + "/",
                protocol_binding="JSONRPC",
            )
        ],
        version=config.AGENT_VERSION,
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
        """
        Emit an event to update task status.

        Args:
            message: The message content to emit
            final: If True, marks the task as complete
            failed: If True, marks the task as failed

        Raises:
            Exception: If event emission fails
        """
        logger.info("Emitting event %s", message)

        if final or failed:
            parts = [new_text_part(message)]
            await self.task_updater.add_artifact(parts)
            if final:
                await self.task_updater.complete()
            if failed:
                await self.task_updater.failed()
        else:
            await self.task_updater.update_status(
                TaskState.TASK_STATE_WORKING,
                self.task_updater.new_agent_message([new_text_part(message)]),
            )


class GenericExecutor(AgentExecutor):
    """
    A class to handle generic assistant execution for A2A Agent.
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        The agent completes tasks through a natural language conversational interface
        """

        # Setup Event Emitter
        task = context.current_task
        if not task:
            task = new_task_from_user_message(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)
        event_emitter = A2AEvent(task_updater)

        user_input = context.get_user_input()
        if not user_input or not user_input.strip():
            await event_emitter.emit_event("Error: Empty input provided", failed=True)
            return

        # Parse Messages
        messages = [HumanMessage(content=user_input)]
        input = {"messages": messages}
        logger.info(f"Processing messages: {input}")

        try:
            output = None
            # Initialize MCP client with error handling
            logger.info(f"Attempting to connect to MCP server(s) at: {config.MCP_URLS}")

            mcpclient = get_mcpclient()

            # Try to get tools to verify connection
            try:
                tools = await mcpclient.get_tools()
                if tools:
                    logger.info(
                        f"Successfully connected to MCP server(s). Available tools: {[tool.name for tool in tools]}"
                    )
                else:
                    logger.warning("No MCP tools available, but agent will continue")
            except Exception as tool_error:
                logger.warning(
                    f"Failed to connect to MCP server(s): {tool_error}. Agent will continue without MCP tools."
                )
                await event_emitter.emit_event(
                    "⚠️ Warning: Cannot connect to MCP server(s). Agent will continue with limited capabilities.",
                    final=False,
                )

            # Create graph (will work even without MCP tools)
            graph = await get_graph(mcpclient)
            async for event in graph.astream(input, stream_mode="updates"):
                await event_emitter.emit_event(
                    "\n".join(
                        f"🚶‍♂️{key}: {str(value)[: config.MAX_EVENT_DISPLAY_LENGTH] + '...' if len(str(value)) > config.MAX_EVENT_DISPLAY_LENGTH else str(value)}"
                        for key, value in event.items()
                    )
                    + "\n"
                )
                output = event
                logger.info(f"event: {event}")

            final_answer = output.get("assistant", {}).get("final_answer") if output else None
            if final_answer is None:
                logger.warning("No final answer received from graph execution")
                await event_emitter.emit_event("Task completed but no final answer was generated.", final=True)
            else:
                await event_emitter.emit_event(str(final_answer), final=True)
        except Exception as e:
            logger.error(f"Graph execution error: {e}")
            await event_emitter.emit_event(f"Error: Failed to process request. {str(e)}", failed=True)
            raise Exception(str(e))

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
        agent_executor=GenericExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    # a2a-sdk 1.x replaced A2AStarletteApplication with route factories that we
    # assemble into a Starlette app ourselves.
    routes = create_jsonrpc_routes(request_handler, rpc_url="/")
    # Serve the current well-known path (/.well-known/agent-card.json) plus the
    # legacy /.well-known/agent.json path for backward compatibility.
    routes += create_agent_card_routes(agent_card)
    routes += create_agent_card_routes(agent_card, card_url="/.well-known/agent.json")

    app = Starlette(routes=routes)

    uvicorn.run(app, host=host, port=port)
