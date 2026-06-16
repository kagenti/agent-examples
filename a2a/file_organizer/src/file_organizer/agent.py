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
from file_organizer.graph import get_graph, get_mcpclient

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

LangChainInstrumentor().instrument()


def get_agent_card(host: str, port: int):
    """Returns the Agent Card for the A2A Agent."""
    capabilities = AgentCapabilities(streaming=True)
    skill = AgentSkill(
        id="file_organizer",
        name="File Organizer",
        description="**File Organizer** – Personalized assistant for organizing files.",
        tags=["file-organization"],
        examples=[
            "Organize all files in this bucket by file type",
            "Move all PDFs to a new folder",
        ],
    )
    return AgentCard(
        name="File Organizer",
        description=dedent(
            """\
            This agent provides a simple file organization assistance.

            ## Input Parameters
            - **prompt** (string) – the action you want to perform on files (move or copy).

            ## Key Features
            - **MCP Tool Calling** – uses a MCP tool to organize files.
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


class FileOrganizerExecutor(AgentExecutor):
    """
    A class to handle file organizer execution for A2A Agent.
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        The agent allows to organize files through a natural language conversational interface
        """

        # Setup Event Emitter
        task = context.current_task
        if not task:
            task = new_task_from_user_message(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)
        event_emitter = A2AEvent(task_updater)

        # Get user input directly
        user_input = context.get_user_input()
        messages = [HumanMessage(content=user_input)]
        input_data = {"messages": messages}
        logger.info(f"Processing messages: {input_data}")

        try:
            # Test MCP connection first
            logger.info(f"Attempting to connect to MCP server at: {os.getenv('MCP_URL', 'http://localhost:8000/sse')}")

            mcpclient = get_mcpclient()

            # Try to get tools to verify connection
            try:
                tools = await mcpclient.get_tools()
                logger.info(f"Successfully connected to MCP server. Available tools: {[tool.name for tool in tools]}")

            except Exception as tool_error:
                logger.error(f"Failed to connect to MCP server: {tool_error}")
                await event_emitter.emit_event(
                    f"Error: Cannot connect to MCP cloud storage at {os.getenv('MCP_URL', 'http://localhost:8000/sse')}. Please ensure the cloud storage MCP server is running. Error: {tool_error}",
                    failed=True,
                )
                return

            graph = await get_graph(mcpclient)
            output = None
            async for event in graph.astream(input_data, stream_mode="updates"):
                await event_emitter.emit_event(
                    "\n".join(
                        f"🚶‍♂️{key}: {str(value)[:256] + '...' if len(str(value)) > 256 else str(value)}"
                        for key, value in event.items()
                    )
                    + "\n"
                )
                output = event
                logger.info(f"event: {event}")

            if output:
                final_answer = output.get("assistant", {}).get("final_answer", "File organization completed.")
                await event_emitter.emit_event(str(final_answer), final=True)
            else:
                await event_emitter.emit_event("File organization completed.", final=True)
        except Exception as e:
            logger.error(f"Graph execution error: {e}")
            await event_emitter.emit_event(
                f"Error: Failed to process file organization request. {str(e)}",
                failed=True,
            )
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
        agent_executor=FileOrganizerExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    routes = []
    routes.extend(create_agent_card_routes(agent_card))
    # enable_v0_3_compat is needed because Kagenti uses A2A 0.3 client libraries
    routes.extend(create_jsonrpc_routes(request_handler, "/", enable_v0_3_compat=True))
    app = Starlette(routes=routes)

    uvicorn.run(app, host=host, port=port)
