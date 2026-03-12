import logging
from textwrap import dedent

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TaskState, TextPart
from a2a.utils import new_agent_text_message, new_task
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

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
        url=f"http://{host}:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
    )


class CheerupExecutor(AgentExecutor):
    """Handles cheerup agent execution for A2A."""

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)

        user_input = context.get_user_input()
        logger.info(
            "Cheerup agent received: %s (context=%s)", user_input, task.context_id
        )

        await task_updater.update_status(
            TaskState.working,
            new_agent_text_message(
                "Brewing up some good vibes...",
                task_updater.context_id,
                task_updater.task_id,
            ),
        )

        try:
            reply = await chat(task.context_id, user_input)

            parts = [TextPart(text=reply)]
            await task_updater.add_artifact(parts)
            await task_updater.update_status(
                TaskState.input_required,
                new_agent_text_message(
                    reply,
                    task_updater.context_id,
                    task_updater.task_id,
                ),
            )
        except Exception as e:
            logger.error("Cheerup agent error: %s", e)
            parts = [TextPart(text="Sorry, something went wrong. Please try again.")]
            await task_updater.add_artifact(parts)
            await task_updater.failed()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def agent_card_compat(request: Request) -> JSONResponse:
    """Serve agent card at /.well-known/agent-card.json for Kagenti backend compatibility."""
    card = get_agent_card(host="0.0.0.0", port=8000)
    return JSONResponse(card.model_dump(mode="json", exclude_none=True))


def run():
    """Runs the A2A Agent application."""
    agent_card = get_agent_card(host="0.0.0.0", port=8000)

    request_handler = DefaultRequestHandler(
        agent_executor=CheerupExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    app = server.build()

    # Add custom routes
    app.routes.insert(0, Route("/health", health, methods=["GET"]))
    app.routes.insert(
        0, Route("/.well-known/agent-card.json", agent_card_compat, methods=["GET"])
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)
