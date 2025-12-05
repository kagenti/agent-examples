import base64
import logging
import os
from textwrap import dedent
import uvicorn

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TaskState, TextPart, DataPart
from a2a.utils import new_agent_text_message, new_task
from openinference.instrumentation.langchain import LangChainInstrumentor
from langchain_core.messages import HumanMessage

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
        examples=["give me a 100x100 image", "show me an image 400x400"],
    )
    return AgentCard(
        name="Image Agent",
        description=dedent(
            """\
            This agent fetches an image from the MCP `image_tool` and returns the base64-encoded
            image (and original URL) back to the UI as a JSON artifact.

            Input: a short text that may include two integers (height width).
            """
        ),
        url=f"http://{host}:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
    )


class ImageEvent:
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


class ImageExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Fetch an image (base64) from the MCP image_tool and return it to the UI."""
        task = context.current_task
        if not task:
            task = new_task(context.message)  
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)
        event_emitter = ImageEvent(task_updater)

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
                await event_emitter.emit_event(f"Error: Cannot connect to MCP image service at {os.getenv('MCP_URL', 'http://localhost:8000/sse')}. Please ensure the image MCP server is running. Error: {tool_error}", failed=True)
                return

            graph = await get_graph(mcpclient)
            messages = [HumanMessage(content=context.get_user_input())]
            input = {"messages": messages}
            output = None
            async for event in graph.astream(input, stream_mode="updates"):
                await event_emitter.emit_event(
                    "\n".join(
                        f"{key}: {str(value)[:100] + '...' if len(str(value)) > 100 else str(value)}"
                        for key, value in event.items()
                    )
                    + "\n"
                )
                output = event
        
            result = output.get("assistant", {}).get("final_answer")

            if not result:
                messages = output.get("assistant", {}).get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "content"):
                        result = last_msg.content
                    elif isinstance(last_msg, dict) and "content" in last_msg:
                        result = last_msg["content"]

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
                        DataPart(
                            data={
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
                await event_emitter.emit_event(str(result), final=True)
                return
            except Exception as e:
                err_msg = f"Error processing graph result: {e}"
                logger.error(err_msg)
                await event_emitter.emit_event(err_msg, failed=True)
                return

        except Exception as e:
            logger.error(f'Graph execution error: {e}')
            await event_emitter.emit_event(f"Error: Failed to process image request. {str(e)}", failed=True)
            # Do not re-raise exception to avoid breaking SSE stream with a 500 response
            return

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Not implemented"""
        raise Exception("cancel not supported")

def run():
    agent_card = get_agent_card(host="0.0.0.0", port=8001)

    request_handler = DefaultRequestHandler(
        agent_executor=ImageExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host="0.0.0.0", port=int(os.getenv("IMAGE_AGENT_PORT", "8001")))