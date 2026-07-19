import logging
import os
from textwrap import dedent

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

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
from cheerup_agent.cheerup_llm import chat

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_agent_card(host: str, port: int):
    """Returns the Agent Card for the Cheerup Agent."""
    capabilities = AgentCapabilities(streaming=False)
    skill = AgentSkill(
        id="cheerup_companion",
        name="Cheerup Companion",
        description="**Cheerup Companion** - A warm, uplifting friend that brightens your day with humor, encouragement, and positive vibes.",
        tags=["cheerful", "motivation", "humor", "wellbeing"],
        examples=[
            "I'm having a rough day",
            "Tell me something to make me smile",
            "I need a pep talk",
            "Make me laugh!",
        ],
    )
    return AgentCard(
        name="Cheerup Companion",
        description=dedent(
            """\
            A cheerful conversational companion designed to brighten your day
            and put you in a great mood.

            ## What I can do
            - Share jokes and fun facts to make you smile
            - Give you a pep talk when you need encouragement
            - Celebrate your wins, big or small
            - Be a positive, supportive friend to chat with
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


class CheerupExecutor(AgentExecutor):
    """Handles cheerup agent execution for A2A."""

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        task = context.current_task
        if not task:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)

        user_input = context.get_user_input()
        logger.info("Cheerup agent received: %s (context=%s)", user_input, task.context_id)

        await task_updater.update_status(
            TaskState.TASK_STATE_WORKING,
            new_text_message(
                "Brewing up some good vibes...",
                context_id=task_updater.context_id,
                task_id=task_updater.task_id,
            ),
        )

        try:
            reply = await chat(task.context_id, user_input)

            parts = [new_text_part(reply)]
            await task_updater.add_artifact(parts)
            await task_updater.update_status(
                TaskState.TASK_STATE_INPUT_REQUIRED,
                new_text_message(
                    reply,
                    context_id=task_updater.context_id,
                    task_id=task_updater.task_id,
                ),
            )
        except Exception as e:
            logger.error("Cheerup agent error: %s", e)
            parts = [new_text_part("Sorry, something went wrong. Please try again.")]
            await task_updater.add_artifact(parts)
            await task_updater.failed()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def run():
    """Runs the A2A Agent application."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    agent_card = get_agent_card(host, port)

    request_handler = DefaultRequestHandler(
        agent_executor=CheerupExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    routes = [Route("/health", health, methods=["GET"])]
    # create_agent_card_routes serves /.well-known/agent-card.json natively
    routes.extend(create_agent_card_routes(agent_card))
    # enable_v0_3_compat is needed because Rossoctl uses A2A 0.3 client libraries
    routes.extend(create_jsonrpc_routes(request_handler, "/", enable_v0_3_compat=True))
    app = Starlette(routes=routes)

    uvicorn.run(app, host=host, port=port)
