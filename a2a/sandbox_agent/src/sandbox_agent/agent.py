"""A2A agent server for the Sandbox Assistant.

Wires together the workspace manager, permission checker, sources config,
and LangGraph graph to serve the A2A protocol over HTTP.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from textwrap import dedent

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TaskState, TextPart
from a2a.utils import new_agent_text_message, new_task
from langchain_core.messages import HumanMessage
from starlette.routing import Route

from sandbox_agent.configuration import Configuration
from sandbox_agent.graph import build_graph
from sandbox_agent.permissions import PermissionChecker
from sandbox_agent.sources import SourcesConfig
from sandbox_agent.workspace import WorkspaceManager

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Package root is two levels up from __file__
# (__file__ = src/sandbox_agent/agent.py -> package root = .)
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_json(filename: str) -> dict:
    """Load a JSON file from the package root directory.

    Parameters
    ----------
    filename:
        Name of the JSON file (e.g. ``settings.json`` or ``sources.json``).

    Returns
    -------
    dict
        Parsed JSON content.
    """
    path = _PACKAGE_ROOT / filename
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Agent Card
# ---------------------------------------------------------------------------


def get_agent_card(host: str, port: int) -> AgentCard:
    """Return an A2A AgentCard for the Sandbox Assistant.

    Parameters
    ----------
    host:
        Hostname or IP address the agent is listening on.
    port:
        Port number the agent is listening on.
    """
    capabilities = AgentCapabilities(streaming=True)
    skill = AgentSkill(
        id="sandbox_assistant",
        name="Sandbox Assistant",
        description=(
            "**Sandbox Assistant** -- Executes shell commands, reads and writes "
            "files in an isolated per-context workspace with permission checks."
        ),
        tags=["shell", "file", "workspace", "sandbox"],
        examples=[
            "Run 'ls -la' in my workspace",
            "Create a Python script that prints hello world",
            "Read the contents of output/results.txt",
        ],
    )
    return AgentCard(
        name="Sandbox Assistant",
        description=dedent(
            """\
            A sandboxed coding assistant that can execute shell commands, \
            read files, and write files inside isolated per-context workspaces.

            ## Key Features
            - **Shell execution** with three-tier permission checks (allow/deny/HITL)
            - **File read/write** with path-traversal prevention
            - **Per-context workspaces** for multi-turn isolation
            """,
        ),
        url=f"http://{host}:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
    )


# ---------------------------------------------------------------------------
# Agent Executor
# ---------------------------------------------------------------------------


class SandboxAgentExecutor(AgentExecutor):
    """A2A executor that delegates to the LangGraph sandbox graph."""

    def __init__(self) -> None:
        settings = _load_json("settings.json")
        sources = _load_json("sources.json")

        self._permission_checker = PermissionChecker(settings)
        self._sources_config = SourcesConfig.from_dict(sources)

        config = Configuration()  # type: ignore[call-arg]
        self._workspace_manager = WorkspaceManager(
            workspace_root=config.workspace_root,
            agent_name="sandbox-assistant",
        )

    # ------------------------------------------------------------------

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Execute a user request through the LangGraph sandbox graph.

        Steps:
        1. Get or create an A2A task.
        2. Resolve the workspace directory from context_id.
        3. Build and stream the LangGraph graph.
        4. Emit status updates and artifacts via TaskUpdater.
        """
        # 1. Get or create task
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)

        task_updater = TaskUpdater(event_queue, task.id, task.context_id)

        # 2. Resolve workspace from context_id
        context_id = task.context_id
        if context_id:
            workspace_path = self._workspace_manager.ensure_workspace(context_id)
            logger.info("Using workspace for context_id=%s: %s", context_id, workspace_path)
        else:
            workspace_path = "/tmp/sandbox-stateless"
            Path(workspace_path).mkdir(parents=True, exist_ok=True)
            logger.info("No context_id; using stateless workspace: %s", workspace_path)

        # 3. Build graph
        graph = build_graph(
            workspace_path=workspace_path,
            permission_checker=self._permission_checker,
            sources_config=self._sources_config,
            checkpointer=None,
        )

        # 4. Stream graph execution
        messages = [HumanMessage(content=context.get_user_input())]
        input_state = {"messages": messages}
        logger.info("Processing messages: %s", input_state)

        try:
            output = None
            async for event in graph.astream(input_state, stream_mode="updates"):
                # Send intermediate status updates
                await task_updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        "\n".join(
                            f"{key}: {str(value)[:256] + '...' if len(str(value)) > 256 else str(value)}"
                            for key, value in event.items()
                        )
                        + "\n",
                        task_updater.context_id,
                        task_updater.task_id,
                    ),
                )
                output = event

            # Extract final answer from the last event
            final_answer = None
            if output:
                # The assistant node returns {"messages": [AIMessage(...)]}
                assistant_output = output.get("assistant", {})
                if isinstance(assistant_output, dict):
                    msgs = assistant_output.get("messages", [])
                    if msgs:
                        final_answer = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])

            if final_answer is None:
                final_answer = "No response generated."

            # Add artifact with final answer and complete
            parts = [TextPart(text=str(final_answer))]
            await task_updater.add_artifact(parts)
            await task_updater.complete()

        except Exception as e:
            logger.error("Graph execution error: %s", e)
            parts = [TextPart(text=f"Error: {e}")]
            await task_updater.add_artifact(parts)
            await task_updater.failed()
            raise

    # ------------------------------------------------------------------

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancel is not supported."""
        raise Exception("cancel not supported")


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Create the A2A server application and run it with uvicorn."""
    agent_card = get_agent_card(host="0.0.0.0", port=8000)

    request_handler = DefaultRequestHandler(
        agent_executor=SandboxAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # Build the Starlette app
    app = server.build()

    # Add the /.well-known/agent-card.json route
    app.routes.insert(
        0,
        Route(
            "/.well-known/agent-card.json",
            server._handle_get_agent_card,
            methods=["GET"],
            name="agent_card_well_known",
        ),
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)
