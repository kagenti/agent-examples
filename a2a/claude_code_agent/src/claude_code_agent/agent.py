import asyncio
import logging
import os
from textwrap import dedent

import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Route

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_task

from claude_code_agent.configuration import Configuration
from claude_code_agent.events import StreamTranslator
from claude_code_agent.runner import run_turn
from claude_code_agent.session import SessionRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_agent_card(host: str, port: int) -> AgentCard:
    capabilities = AgentCapabilities(streaming=True)
    skill = AgentSkill(
        id="claude_code",
        name="Claude Code",
        description="General-purpose coding and reasoning assistant powered by Claude Code.",
        tags=["coding", "claude-code", "agent"],
        examples=[
            "Write a Python function that reverses a linked list.",
            "Explain what this repository does.",
            "Create a file hello.txt containing 'hi'.",
        ],
    )
    return AgentCard(
        name="Claude Code Agent",
        description=dedent(
            """\
            An agent that drives Claude Code headlessly. Each A2A session maps to a
            persistent Claude Code conversation with its own isolated workspace, so
            concurrent sessions (same or multiple users) stay separate.

            ## Input Parameters
            - **prompt** (string) – the instruction or question for Claude Code.
            """
        ),
        url=os.getenv("AGENT_ENDPOINT", f"http://{host}:{port}").rstrip("/") + "/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
    )


class ClaudeCodeExecutor(AgentExecutor):
    """Runs one Claude Code turn per A2A message, isolated per context_id."""

    def __init__(
        self,
        config: Configuration,
        registry: SessionRegistry,
        semaphore: asyncio.Semaphore,
    ):
        self._config = config
        self._registry = registry
        self._semaphore = semaphore

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)
        translator = StreamTranslator(task_updater)

        prompt = context.get_user_input()
        session = await self._registry.get_or_create(task.context_id)

        # Serialize turns within a session first (so queued same-session turns
        # don't consume a global slot), then bound total concurrency.
        async with session.lock:
            async with self._semaphore:
                try:
                    await run_turn(session, prompt, translator, self._config)
                except Exception as exc:  # noqa: BLE001 - surface any failure to A2A
                    logger.exception("turn failed")
                    translator.errored = True
                    translator.error_reason = str(exc)

        await translator.finish()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")


def run() -> None:
    config = Configuration()
    if not config.has_auth_token:
        logger.warning(
            "ANTHROPIC_AUTH_TOKEN is not set; Claude Code calls to the LiteLLM "
            "endpoint will fail until it is provided."
        )

    registry = SessionRegistry(config.workspace_root, config.max_sessions)
    semaphore = asyncio.Semaphore(config.max_concurrent)

    agent_card = get_agent_card(host=config.host, port=config.port)
    request_handler = DefaultRequestHandler(
        agent_executor=ClaudeCodeExecutor(config, registry, semaphore),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
    app = server.build()

    # Serve the newer agent-card.json path alongside the legacy agent.json path.
    app.routes.insert(
        0,
        Route(
            "/.well-known/agent-card.json",
            server._handle_get_agent_card,
            methods=["GET"],
            name="agent_card_new",
        ),
    )

    uvicorn.run(app, host=config.host, port=config.port)
