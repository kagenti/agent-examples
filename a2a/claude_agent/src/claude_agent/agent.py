import asyncio
import logging
import os
from textwrap import dedent

import uvicorn
from starlette.applications import Starlette

from a2a.helpers import new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from claude_agent.configuration import Configuration
from claude_agent.events import StreamTranslator
from claude_agent.runner import run_turn
from claude_agent.session import SessionRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_agent_card(host: str, port: int) -> AgentCard:
    capabilities = AgentCapabilities(streaming=True)
    skill = AgentSkill(
        id="claude_agent",
        name="Claude Agent",
        description="General-purpose coding and reasoning assistant powered by Claude.",
        tags=["coding", "claude", "agent"],
        examples=[
            "Write a Python function that reverses a linked list.",
            "Explain what this repository does.",
            "Create a file hello.txt containing 'hi'.",
        ],
    )
    return AgentCard(
        name="Claude Agent",
        description=dedent(
            """\
            An agent that drives Claude headlessly. Each A2A session maps to a
            persistent Claude conversation with its own isolated workspace, so
            concurrent sessions (same or multiple users) stay separate.

            ## Input Parameters
            - **prompt** (string) – the instruction or question for Claude.
            """
        ),
        supported_interfaces=[
            AgentInterface(
                url=os.getenv("AGENT_ENDPOINT", f"http://{host}:{port}").rstrip("/") + "/",
                protocol_binding="JSONRPC",
            )
        ],
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
    )


class ClaudeAgentExecutor(AgentExecutor):
    """Runs one Claude turn per A2A message, isolated per context_id."""

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
            task = new_task_from_user_message(context.message)  # type: ignore
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
        # A2A cancel is not supported. Note: if the executing turn's coroutine is
        # cancelled (e.g. client disconnect), run_turn() kills the subprocess in
        # its finally block, so no claude process is orphaned.
        raise NotImplementedError("cancel not supported")


def run() -> None:
    config = Configuration()
    if not config.has_auth_token:
        logger.warning(
            "ANTHROPIC_AUTH_TOKEN is not set; Claude calls to the LiteLLM endpoint will fail until it is provided."
        )
    if not config.anthropic_base_url:
        logger.warning(
            "ANTHROPIC_BASE_URL is not set; Claude will use api.anthropic.com. "
            "Set it to your LiteLLM endpoint to route through the gateway."
        )

    registry = SessionRegistry(config.workspace_root, config.max_sessions)
    semaphore = asyncio.Semaphore(config.max_concurrent)

    agent_card = get_agent_card(host=config.host, port=config.port)
    request_handler = DefaultRequestHandler(
        agent_executor=ClaudeAgentExecutor(config, registry, semaphore),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    # a2a-sdk 1.x replaced A2AStarletteApplication with route factories that we
    # assemble into a Starlette app ourselves. Serve the current well-known path
    # (/.well-known/agent-card.json) plus the legacy /.well-known/agent.json for
    # back-compat.
    # enable_v0_3_compat is needed because Rossoctl uses A2A 0.3 client libraries
    routes = create_jsonrpc_routes(request_handler, rpc_url="/", enable_v0_3_compat=True)
    routes += create_agent_card_routes(agent_card)
    routes += create_agent_card_routes(agent_card, card_url="/.well-known/agent.json")
    app = Starlette(routes=routes)

    uvicorn.run(app, host=config.host, port=config.port)
