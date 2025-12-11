import logging
import os
import uvicorn
from textwrap import dedent

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TaskState, TextPart
from a2a.utils import new_agent_text_message, new_task
from openinference.instrumentation.langchain import LangChainInstrumentor
from langchain_core.messages import HumanMessage

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
        url=f"http://{host}:{port}/",
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
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)
        event_emitter = A2AEvent(task_updater)

        # Parse Messages
        messages = [HumanMessage(content=context.get_user_input())]
        input = {"messages": messages}
        logger.info(f'Processing messages: {input}')

        task_updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            output = None
            # Test MCP connection first
            mcp_url = os.getenv("MCP_URL", "http://reservation-tool:8000/mcp")
            logger.info(f'Attempting to connect to MCP server at: {mcp_url}')

            mcpclient = get_mcpclient()

            # Try to get tools to verify connection
            try:
                tools = await mcpclient.get_tools()
                logger.info(f'Successfully connected to MCP server. Available tools: {[tool.name for tool in tools]}')
            except Exception as tool_error:
                logger.error(f'Failed to connect to MCP server: {tool_error}')
                await event_emitter.emit_event(
                    f"Error: Cannot connect to reservation MCP service at {mcp_url}. "
                    f"Please ensure the reservation MCP server is running. Error: {tool_error}",
                    failed=True
                )
                return

            graph = await get_graph(mcpclient)
            async for event in graph.astream(input, stream_mode="updates"):
                await event_emitter.emit_event(
                    "\n".join(
                        f"🤔 {key}: {str(value)[:100] + '...' if len(str(value)) > 100 else str(value)}"
                        for key, value in event.items()
                    )
                    + "\n"
                )
                output = event
                logger.info(f'event: {event}')
            output = output.get("assistant", {}).get("final_answer")
            await event_emitter.emit_event(str(output), final=True)
        except Exception as e:
            logger.error(f'Graph execution error: {e}')
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
    agent_card = get_agent_card(host="0.0.0.0", port=8000)

    request_handler = DefaultRequestHandler(
        agent_executor=ReservationExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    uvicorn.run(server.build(), host="0.0.0.0", port=8000)
