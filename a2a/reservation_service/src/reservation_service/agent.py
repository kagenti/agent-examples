import logging
import os
from textwrap import dedent

import uvicorn
from langchain_core.messages import HumanMessage
from openinference.instrumentation.langchain import LangChainInstrumentor
from starlette.applications import Starlette

from a2a.helpers import (
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskState,
)
from reservation_service.graph import get_graph, get_mcpclient

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

LangChainInstrumentor().instrument()


def get_agent_card(host: str, port: int):
    """Returns the Agent Card for the Reservation Agent."""
    capabilities = AgentCapabilities(streaming=True)
    skill = AgentSkill(
        id="reservation_assistant",
        name="Reservation Assistant",
        description="**Reservation Assistant** – AI-powered restaurant reservation helper.",
        tags=["reservations", "restaurants", "dining"],
        examples=[
            "Find Italian restaurants in Boston",
            "Check availability at Trattoria di Mare for 4 people on December 25th at 7 PM",
            "Make a reservation at Trattoria di Mare for tonight at 7 PM, party of 2",
            "List my reservations for john@example.com",
            "Cancel my reservation",
        ],
    )
    return AgentCard(
        name="Reservation Assistant",
        description=dedent(
            """\
            This agent provides restaurant reservation assistance through natural language conversation.

            ## Capabilities
            - **Search Restaurants** – Find restaurants by city, cuisine, and price tier
            - **Check Availability** – See available time slots at specific restaurants
            - **Make Reservations** – Book tables with confirmation codes
            - **Manage Reservations** – List and cancel existing reservations
            - **Conversational Interface** – Natural language interaction

            ## Input Parameters
            - **prompt** (string) – Your request in natural language

            ## Key Features
            - **MCP Tool Calling** – Connects to restaurant reservation MCP tools
            - **Multi-step Workflows** – Handles complex reservation scenarios
            - **Real-time Availability** – Checks current restaurant availability
            """,
        ),
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
        supported_interfaces=[
            AgentInterface(
                # Allow env var AGENT_ENDPOINT to override the URL in the agent card
                url=os.getenv("AGENT_ENDPOINT", f"http://{host}:{port}").rstrip("/") + "/",
                protocol_binding="JSONRPC",
            )
        ],
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
            parts = [new_text_part(message)]
            await self.task_updater.add_artifact(parts)
            if final:
                await self.task_updater.complete()
            if failed:
                await self.task_updater.failed()
        else:
            await self.task_updater.update_status(
                TaskState.TASK_STATE_WORKING,
                new_text_message(
                    message,
                    context_id=self.task_updater.context_id,
                    task_id=self.task_updater.task_id,
                ),
            )


class ReservationExecutor(AgentExecutor):
    """
    A class to handle reservation assistant execution for A2A Agent.
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        The agent allows restaurant reservations through a natural language conversational interface
        """

        # Setup Event Emitter
        task = context.current_task
        if not task:
            task = new_task_from_user_message(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)
        event_emitter = A2AEvent(task_updater)

        # Parse Messages
        messages = [HumanMessage(content=context.get_user_input())]
        input = {"messages": messages}
        logger.info(f"Processing messages: {input}")

        try:
            output = None
            # Test MCP connection first
            mcp_url = os.getenv("MCP_URL", "http://reservation-tool:8000/mcp")
            logger.info(f"Attempting to connect to MCP server at: {mcp_url}")

            mcpclient = get_mcpclient()

            # Try to get tools to verify connection
            try:
                tools = await mcpclient.get_tools()
                logger.info(f"Successfully connected to MCP server. Available tools: {[tool.name for tool in tools]}")
            except Exception as tool_error:
                logger.error(f"Failed to connect to MCP server: {tool_error}")
                await event_emitter.emit_event(
                    f"Error: Cannot connect to reservation MCP service at {mcp_url}. "
                    f"Please ensure the reservation MCP server is running. Error: {tool_error}",
                    failed=True,
                )
                return

            graph = await get_graph(mcpclient)
            async for event in graph.astream(input, stream_mode="updates"):
                await event_emitter.emit_event(
                    "\n".join(
                        f"🤔 {key}: {str(value)[:256] + '...' if len(str(value)) > 256 else str(value)}"
                        for key, value in event.items()
                    )
                    + "\n"
                )
                output = event
                logger.info(f"event: {event}")
            if output is not None:
                final_answer = output.get("assistant", {}).get("final_answer")
                await event_emitter.emit_event(str(final_answer), final=True)
            else:
                await event_emitter.emit_event("No events produced by the graph.", final=True)
        except Exception as e:
            logger.error(f"Graph execution error: {e}")
            await event_emitter.emit_event(f"Error: Failed to process reservation request. {str(e)}", failed=True)
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
    agent_card = get_agent_card(host, port)

    request_handler = DefaultRequestHandler(
        agent_executor=ReservationExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    routes = []
    routes.extend(create_agent_card_routes(agent_card))
    # enable_v0_3_compat is needed because Rossoctl uses A2A 0.3 client libraries
    routes.extend(create_jsonrpc_routes(request_handler, "/", enable_v0_3_compat=True))
    app = Starlette(routes=routes)

    uvicorn.run(app, host=host, port=port)
