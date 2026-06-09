import argparse
import base64
import logging
import os
from textwrap import dedent

import uvicorn
from langchain_core.messages import HumanMessage
from openinference.instrumentation.langchain import LangChainInstrumentor
from starlette.applications import Starlette

from a2a.helpers import (
    new_data_part,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskState,
)
from image_service.graph import get_graph, get_mcpclient

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

LangChainInstrumentor().instrument()


def get_agent_card(host: str, port: int):
    """Returns the Agent Card for the Image Agent."""
    capabilities = AgentCapabilities(streaming=True)
    skill = AgentSkill(
        id="image_agent",
        name="Image Agent",
        description="Agent that requests an image from the image MCP tool and returns the base64 to the UI.",
        tags=["image"],
        examples=["give me a 100x100 image", "show me an image that is 400 by 400"],
    )
    # Allow env var AGENT_ENDPOINT to override the URL in the agent card
    url = os.getenv("AGENT_ENDPOINT", f"http://{host}:{port}").rstrip("/") + "/"
    return AgentCard(
        name="Image Agent",
        description=dedent(
            """\
            This agent fetches an image from the MCP `image_tool` and returns the base64-encoded
            image (and original URL) back to the UI as a JSON artifact.

            Input: a short text that may include two integers (height width).
            """
        ),
        supported_interfaces=[
            AgentInterface(url=url, protocol_binding="JSONRPC"),
        ],
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
    )


class ImageTaskEventEmitter:
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


class ImageExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Fetch an image (base64) from the MCP image_tool and return it to the UI."""
        task = context.current_task
        if not task:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)
        event_emitter = ImageTaskEventEmitter(task_updater)

        try:
            # Test MCP connection first
            logger.info(
                "Attempting to connect to MCP server at: %s",
                os.getenv("MCP_URL", "http://localhost:8000/mcp"),
            )
            mcpclient = get_mcpclient()
            # Try to get tools to verify connection
            try:
                tools = await mcpclient.get_tools()
                logger.info(
                    "Successfully connected to MCP server. Available tools: %s",
                    [tool.name for tool in tools],
                )
            except Exception as tool_error:
                logger.error("Failed to connect to MCP server: %s", tool_error)
                await event_emitter.emit_event(
                    f"Error: Cannot connect to MCP image service at {os.getenv('MCP_URL', 'http://localhost:8000/mcp')}. Please ensure the image MCP server is running. Error: {tool_error}",
                    failed=True,
                )
                return

            graph = await get_graph(mcpclient)
            messages = [HumanMessage(content=context.get_user_input())]
            graph_input = {"messages": messages}
            output = None
            async for event in graph.astream(graph_input, stream_mode="updates"):
                await event_emitter.emit_event(
                    "\n".join(
                        f"{key}: {str(value)[:256] + '...' if len(str(value)) > 256 else str(value)}"
                        for key, value in event.items()
                    )
                    + "\n"
                )
                output = event

                if output is None:
                    err_msg = "No events were produced by the graph stream; cannot process result."
                    logger.error(err_msg)
                    await event_emitter.emit_event(err_msg, failed=True)
                    return

            result = output.get("assistant", {}).get("final_answer")

            if not result:
                messages = output.get("assistant", {}).get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "content"):
                        result = last_msg.content
                    elif isinstance(last_msg, dict) and "content" in last_msg:
                        result = last_msg["content"]
                    else:
                        result = ""
                else:
                    result = ""

            try:
                # Check if it looks like our image result structure
                if isinstance(result, dict) and "image_base64" in result:
                    image_base64 = result.get("image_base64")
                    image_url = result.get("url")

                    if isinstance(image_base64, (bytes, bytearray)):
                        content_b64 = base64.b64encode(image_base64).decode("utf-8")
                    else:
                        content_b64 = str(image_base64)

                    parts = [
                        new_data_part(
                            {
                                "content": content_b64,
                                "content_encoding": "base64",
                                "content_type": "image/png",
                                "source_url": image_url,
                            }
                        )
                    ]

                    await task_updater.add_artifact(parts, name="image.png")
                    await task_updater.complete()
                    return

                # Fallback: treat as text
                if not result or (isinstance(result, str) and result.strip() == ""):
                    await event_emitter.emit_event(
                        "I am here to help with image requests. Please ask for an image with specific dimensions.",
                        final=True,
                    )
                else:
                    await event_emitter.emit_event(str(result), final=True)
                return
            except Exception as e:
                err_msg = f"Error processing graph result: {e}"
                await event_emitter.emit_event(err_msg, failed=True)
                return

        except Exception as e:
            logger.exception("Graph execution error")
            await event_emitter.emit_event(
                f"Error: Failed to process image request. {type(e).__name__}: {str(e)}",
                failed=True,
            )
            return

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Not implemented"""
        raise NotImplementedError("cancel not supported")


def run():
    parser = argparse.ArgumentParser(description="Run the Image Agent A2A server.")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()
    host, port = args.host, args.port

    agent_card = get_agent_card(host=host, port=port)

    request_handler = DefaultRequestHandler(
        agent_executor=ImageExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    # A2AStarletteApplication was removed in a2a-sdk 1.x; build the Starlette
    # app directly from the route factories. The agent card is served at the
    # current /.well-known/agent-card.json path.
    routes = create_agent_card_routes(agent_card)
    routes += create_jsonrpc_routes(request_handler, rpc_url="/")

    app = Starlette(routes=routes)

    uvicorn.run(app, host=host, port=port)
