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

from weather_service.graph import get_graph, get_mcpclient

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

LangChainInstrumentor().instrument()


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

        # Parse Messages
        messages = [HumanMessage(content=context.get_user_input())]
        input = {"messages": messages}
        logger.info(f'Processing messages: {input}')

        task_updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            output = None
            # Test MCP connection first
            logger.info(f'Attempting to connect to MCP server at: {os.getenv("MCP_URL", "http://localhost:8000/sse")}')

            mcpclient = get_mcpclient()

            # Try to get tools to verify connection
            try:
                tools = await mcpclient.get_tools()
                logger.info(f'Successfully connected to MCP server. Available tools: {[tool.name for tool in tools]}')
            except Exception as tool_error:
                logger.error(f'Failed to connect to MCP server: {tool_error}')
                await event_emitter.emit_event(f"Error: Cannot connect to MCP weather service at {os.getenv('MCP_URL', 'http://localhost:8000/sse')}. Please ensure the weather MCP server is running. Error: {tool_error}", failed=True)
                return

            graph = await get_graph(mcpclient)
            async for event in graph.astream(input, stream_mode="updates"):
                await event_emitter.emit_event(
                    "\n".join(
                        f"🚶‍♂️{key}: {str(value)[:100] + '...' if len(str(value)) > 100 else str(value)}"
                        for key, value in event.items()
                    )
                    + "\n"
                )
                output = event
                logger.info(f'event: {event}')
            output =  output.get("assistant", {}).get("final_answer")
            await event_emitter.emit_event(str(output), final=True)
        except Exception as e:
            logger.error(f'Graph execution error: {e}')
            await event_emitter.emit_event(f"Error: Failed to process weather request. {str(e)}", failed=True)
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
        agent_executor=WeatherExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    uvicorn.run(server.build(), host="0.0.0.0", port=8000)
