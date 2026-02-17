"""A2A server implementation."""

import logging
import traceback
from typing import Any
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message, new_task

from autogen.mcp.mcp_client import create_toolkit, Toolkit
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from simple_generalist.config import Settings
from simple_generalist.agent import GeneralistAgent

logger = logging.getLogger(__name__)


def get_agent_card(settings: Settings) -> AgentCard:
    """
    Create the Agent Card for the Simple Generalist Agent.
    
    Args:
        settings: Application settings
        
    Returns:
        AgentCard describing the agent's capabilities
    """
    capabilities = AgentCapabilities(streaming=True)
    
    # Create skill description
    skill_description = (
        "A general-purpose agent that can use MCP tools to accomplish various tasks. "
        "The agent uses a function-calling loop to iteratively solve problems by calling tools "
        "and synthesizing results."
    )
    
    skill = AgentSkill(
        id="simple_generalist_agent",
        name="Simple Generalist Agent",
        description=skill_description,
        tags=["general", "mcp", "tools", "react", "function-calling"],
        examples=[
            "Help me accomplish tasks using available tools",
            "Search for information and provide insights",
            "Perform multi-step operations with tool assistance",
        ],
    )
    
    agent_url = settings.A2A_PUBLIC_URL
    if not agent_url:
        if settings.A2A_HOST == "0.0.0.0":
            agent_url = f"http://localhost:{settings.A2A_PORT}/"
        else:
            agent_url = f"http://{settings.A2A_HOST}:{settings.A2A_PORT}/"

    return AgentCard(
        name="Simple Generalist Agent",
        description="A generalist AG2 agent exposed via A2A, powered by MCP tools",
        url=agent_url,
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
    )


class SimpleGeneralistExecutor(AgentExecutor):
    """Agent executor for the Simple Generalist Agent."""
    
    def __init__(self, settings: Settings):
        """
        Initialize the executor.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
    
    async def _run_agent(
        self,
        user_input: str,
        settings: Settings,
        event_callback: Any,
        toolkit: Toolkit | None,
    ):
        """Run the agent with the given toolkit."""
        agent = GeneralistAgent(
            settings=settings,
            mcp_toolkit=toolkit,
            event_callback=event_callback,
        )
        
        result = await agent.run_task(user_input)
        
        # Send final result
        final_message = result.get("answer", "Task completed")
        await event_callback(final_message, final=True)

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        Execute a task request.
        
        Args:
            context: Request context
            event_queue: Event queue for progress updates
        """
        # Get or create task
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        
        # Create task updater for progress events
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        # Create event callback
        async def event_callback(message: str, final: bool = False):
            """Send progress events to the client."""
            logger.info(f"Event: {message} (final={final})")
            
            if final:
                # Final message with artifact
                parts = [Part(root=TextPart(text=message))]
                await task_updater.add_artifact(parts)
                await task_updater.complete()
            else:
                # Progress update
                await task_updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        message,
                        task_updater.context_id,
                        task_updater.task_id,
                    ),
                )
        
        # Extract user input
        user_input = context.get_user_input()
        logger.info(f"Processing request: {user_input}")
        
        # Hook up MCP tools (per-request connection like in a2a_agent.py)
        toolkit = None
        try:
            mcp_url = self.settings.MCP_SERVER_URL.strip()
            if mcp_url:
                logger.info(f"Connecting to MCP server at {mcp_url}")

                async with streamablehttp_client(
                    url=mcp_url,
                ) as (
                    read_stream,
                    write_stream,
                    _,
                ), ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    toolkit = await create_toolkit(
                        session=session, use_mcp_resources=False
                    )
                    await self._run_agent(
                        user_input,
                        self.settings,
                        event_callback,
                        toolkit,
                    )
            else:
                await self._run_agent(
                    user_input,
                    self.settings,
                    event_callback,
                    toolkit,
                )
                
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"Error executing task: {exc}", exc_info=True)
            error_message = f"I encountered an error while processing your request: {str(exc)}"
            await event_callback(error_message, final=True)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Cancel a task (not implemented).
        
        Args:
            context: Request context
            event_queue: Event queue
        """
        raise NotImplementedError("Task cancellation is not supported")


def create_app(settings: Settings) -> Any:
    """
    Create the A2A Starlette application.
    
    Args:
        settings: Application settings
        
    Returns:
        Starlette application
    """
    # Create agent card
    agent_card = get_agent_card(settings)
    
    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=SimpleGeneralistExecutor(settings),
        task_store=InMemoryTaskStore(),
    )
    
    # Create A2A server
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    # Build and return the app
    app = server.build()
    logger.info("A2A server application created")
    
    return app

# Made with Bob
