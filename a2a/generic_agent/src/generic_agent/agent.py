import logging
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

from generic_agent.graph import get_graph, get_mcpclient, get_mcp_server_names
from generic_agent.config import Configuration

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
    mcp_section = ""
    if mcp_names:
        mcp_section = "\n\nConnected MCP Servers:\n" + "\n".join(f"- {name}" for name in mcp_names)
    
    
    capabilities = AgentCapabilities(streaming=True)
    skill = AgentSkill(
        id="generic_agent",
        name="Generic Agent",
        description="**Generic Assistant** – Multi-purpose assistant for different tasks based on different MCP tools.",
        tags=mcp_names,
        examples=[],
    )
    return AgentCard(
        name="Generic Agent",
        description=dedent(
            f"""\
            This agent provides assistance for various tasks using different MCP tools.{mcp_section}
            """,
        ),
        url=f"http://{host}:{port}/",
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
            task = new_task(context.message)  # type: ignore
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
        logger.info(f'Processing messages: {input}')

        try:
            output = None
            # Test MCP connection first
            logger.info(f'Attempting to connect to MCP server(s) at: {config.MCP_URLS}')

            mcpclient = get_mcpclient()

            # Try to get tools to verify connection
            try:
                tools = await mcpclient.get_tools()
                logger.info(f'Successfully connected to MCP server(s). Available tools: {[tool.name for tool in tools]}')
            except Exception as tool_error:
                logger.error(f'Failed to connect to MCP server(s): {tool_error}')
                await event_emitter.emit_event(f"Error: Cannot connect to MCP server(s) at {config.MCP_URLS}. Please ensure the MCP server(s) are running. Error: {tool_error}", failed=True)
                return

            graph = await get_graph(mcpclient)
            async for event in graph.astream(input, stream_mode="updates"):
                await event_emitter.emit_event(
                    "\n".join(
                        f"🚶‍♂️{key}: {str(value)[:config.MAX_EVENT_DISPLAY_LENGTH] + '...' if len(str(value)) > config.MAX_EVENT_DISPLAY_LENGTH else str(value)}"
                        for key, value in event.items()
                    )
                    + "\n"
                )
                output = event
                logger.info(f'event: {event}')

            final_answer = output.get("assistant", {}).get("final_answer") if output else None
            if final_answer is None:
                logger.warning("No final answer received from graph execution")
                await event_emitter.emit_event("Task completed but no final answer was generated.", final=True)
            else:
                await event_emitter.emit_event(str(final_answer), final=True)
        except Exception as e:
            logger.error(f'Graph execution error: {e}')
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
    agent_card = get_agent_card(host="0.0.0.0", port=8000)

    request_handler = DefaultRequestHandler(
        agent_executor=GenericExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    uvicorn.run(server.build(), host="0.0.0.0", port=8000)
