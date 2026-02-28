"""A2A agent server for the Sandbox Legion.

Wires together the workspace manager, permission checker, sources config,
and LangGraph graph to serve the A2A protocol over HTTP.
"""

from __future__ import annotations

import asyncio
import hashlib
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

try:
    from a2a.server.tasks import DatabaseTaskStore

    _HAS_SQL_STORE = True
except ImportError:
    _HAS_SQL_STORE = False
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TaskState, TextPart
from a2a.utils import new_agent_text_message, new_task
from langchain_core.messages import HumanMessage
from starlette.routing import Route

from langgraph.checkpoint.memory import MemorySaver

from sandbox_agent.configuration import Configuration
from sandbox_agent.event_serializer import LangGraphSerializer
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
# TOFU (Trust-On-First-Use) verification
# ---------------------------------------------------------------------------

_TOFU_HASH_FILE = ".tofu-hashes.json"

# Files in the workspace root to track for TOFU verification.
_TOFU_TRACKED_FILES = ("CLAUDE.md", "sources.json", "settings.json")


def _hash_file(path: Path) -> str | None:
    """Return the SHA-256 hex digest of a file, or None if it doesn't exist."""
    if not path.is_file():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _compute_tofu_hashes(root: Path) -> dict[str, str]:
    """Compute SHA-256 hashes for tracked files under *root*.

    Returns a dict mapping filename -> hex digest (only for files that exist).
    """
    hashes: dict[str, str] = {}
    for name in _TOFU_TRACKED_FILES:
        digest = _hash_file(root / name)
        if digest is not None:
            hashes[name] = digest
    return hashes


def _tofu_verify(root: Path) -> None:
    """Run TOFU verification on startup.

    On first run, computes and stores hashes of tracked files.  On subsequent
    runs, compares current hashes against the stored ones and logs a WARNING
    if any file has changed (possible tampering).  Does NOT block startup.
    """
    hash_file = root / _TOFU_HASH_FILE
    current_hashes = _compute_tofu_hashes(root)

    if not current_hashes:
        logger.info("TOFU: no tracked files found in %s; skipping.", root)
        return

    if hash_file.is_file():
        try:
            with open(hash_file, encoding="utf-8") as fh:
                stored_hashes = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("TOFU: could not read %s: %s", hash_file, exc)
            stored_hashes = {}

        # Compare each tracked file.
        changed: list[str] = []
        added: list[str] = []
        removed: list[str] = []
        for name, digest in current_hashes.items():
            stored = stored_hashes.get(name)
            if stored is None:
                added.append(name)
            elif stored != digest:
                changed.append(name)
        for name in stored_hashes:
            if name not in current_hashes:
                removed.append(name)

        if changed or added or removed:
            logger.warning(
                "TOFU: workspace file integrity mismatch! "
                "changed=%s, added=%s, removed=%s. "
                "This may indicate tampering. Updating stored hashes.",
                changed, added, removed,
            )
            # Update stored hashes (trust the new state).
            with open(hash_file, "w", encoding="utf-8") as fh:
                json.dump(current_hashes, fh, indent=2)
        else:
            logger.info("TOFU: all tracked files match stored hashes.")
    else:
        # First run: store hashes.
        logger.info("TOFU: first run -- storing hashes for %s", list(current_hashes.keys()))
        with open(hash_file, "w", encoding="utf-8") as fh:
            json.dump(current_hashes, fh, indent=2)


# ---------------------------------------------------------------------------
# Agent Card
# ---------------------------------------------------------------------------


def get_agent_card(host: str, port: int) -> AgentCard:
    """Return an A2A AgentCard for the Sandbox Legion.

    Parameters
    ----------
    host:
        Hostname or IP address the agent is listening on.
    port:
        Port number the agent is listening on.
    """
    capabilities = AgentCapabilities(streaming=True)
    skill = AgentSkill(
        id="sandbox_legion",
        name="Sandbox Legion",
        description=(
            "**Sandbox Legion** -- Executes shell commands, reads and writes "
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
        name="Sandbox Legion",
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

    # Per-context_id locks to serialize concurrent graph executions for the
    # same conversation.  A simple dict + mutex approach with periodic cleanup
    # of unused entries.
    _context_locks: dict[str, asyncio.Lock] = {}
    _context_locks_mutex: asyncio.Lock = asyncio.Lock()

    async def _get_context_lock(self, context_id: str) -> asyncio.Lock:
        """Return (and lazily create) the asyncio.Lock for *context_id*.

        A class-level mutex guards the dict so that two concurrent requests
        for the same new context_id don't each create their own Lock.
        """
        async with self._context_locks_mutex:
            lock = self._context_locks.get(context_id)
            if lock is None:
                lock = asyncio.Lock()
                self._context_locks[context_id] = lock
            return lock

    def __init__(self) -> None:
        settings = _load_json("settings.json")
        sources = _load_json("sources.json")

        self._permission_checker = PermissionChecker(settings)
        self._sources_config = SourcesConfig.from_dict(sources)

        config = Configuration()  # type: ignore[call-arg]

        # Use PostgreSQL checkpointer if configured, else in-memory
        self._checkpoint_db_url = config.checkpoint_db_url
        self._checkpointer = None  # Lazy-initialized in execute()
        self._checkpointer_initialized = False
        if not self._checkpoint_db_url or self._checkpoint_db_url == "memory":
            self._checkpointer = MemorySaver()
            self._checkpointer_initialized = True
            logger.info("Using in-memory checkpointer (set CHECKPOINT_DB_URL for persistence)")
        else:
            logger.info("PostgreSQL checkpointer configured: %s", self._checkpoint_db_url.split("@")[-1])
        self._workspace_manager = WorkspaceManager(
            workspace_root=config.workspace_root,
            agent_name="sandbox-legion",
            ttl_days=config.context_ttl_days,
        )

        # C19: Clean up expired workspaces on startup.
        cleaned = self._workspace_manager.cleanup_expired()
        if cleaned:
            logger.info("Cleaned up %d expired workspaces: %s", len(cleaned), cleaned)

        # TOFU: verify workspace config file integrity on startup.
        # Logs warnings on mismatch but does not block the agent from starting.
        _tofu_verify(_PACKAGE_ROOT)

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

        # Lazy-init PostgreSQL checkpointer on first execute()
        if not self._checkpointer_initialized and self._checkpoint_db_url:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            # from_conn_string returns a context manager; enter it and keep
            # the saver alive for the process lifetime.
            cm = AsyncPostgresSaver.from_conn_string(self._checkpoint_db_url)
            self._checkpointer = await cm.__aenter__()
            self._checkpointer_cm = cm  # prevent GC
            await self._checkpointer.setup()
            self._checkpointer_initialized = True
            logger.info("PostgreSQL checkpointer initialized")

        # 3. Build graph with shared checkpointer for multi-turn memory
        graph = build_graph(
            workspace_path=workspace_path,
            permission_checker=self._permission_checker,
            sources_config=self._sources_config,
            checkpointer=self._checkpointer,
        )

        # 4. Stream graph execution with thread_id for checkpointer routing.
        #    Acquire a per-context_id lock so that two concurrent requests for
        #    the same conversation are serialized (the LangGraph checkpointer
        #    is not safe for parallel writes to the same thread_id).
        lock = await self._get_context_lock(context_id or "stateless")
        logger.info(
            "Acquiring context lock for context_id=%s (already locked: %s)",
            context_id,
            lock.locked(),
        )

        async with lock:
            messages = [HumanMessage(content=context.get_user_input())]
            input_state = {"messages": messages}
            graph_config = {"configurable": {"thread_id": context_id or "stateless"}}
            logger.info("Processing messages: %s (thread_id=%s)", input_state, context_id)

            try:
                output = None
                serializer = LangGraphSerializer()
                async for event in graph.astream(input_state, config=graph_config, stream_mode="updates"):
                    # Send intermediate status updates as structured JSON
                    await task_updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            "\n".join(
                                serializer.serialize(key, value)
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
                            content = getattr(msgs[-1], "content", None)
                            if isinstance(content, list):
                                # Tool-calling models return a list of content blocks;
                                # extract only the text portions.
                                final_answer = "\n".join(
                                    block.get("text", "") if isinstance(block, dict) else str(block)
                                    for block in content
                                    if isinstance(block, dict) and block.get("type") == "text"
                                ) or None
                            elif content:
                                final_answer = str(content)

                if final_answer is None:
                    final_answer = "No response generated."

                # Add artifact with final answer and complete
                parts = [TextPart(text=final_answer)]
                await task_updater.add_artifact(parts)
                await task_updater.complete()

            except Exception as e:
                logger.error("Graph execution error: %s", e)
                parts = [TextPart(text=f"Error: {e}")]
                await task_updater.add_artifact(parts)
                await task_updater.failed()
                raise

        # Periodic cleanup: remove locks that are no longer held and whose
        # context_id has not been seen recently.  We do this opportunistically
        # after each execution to avoid unbounded growth.
        async with self._context_locks_mutex:
            stale = [cid for cid, lk in self._context_locks.items() if not lk.locked()]
            # Keep the dict from growing without bound, but only drop entries
            # when there are more than 1000 idle locks.
            if len(stale) > 1000:
                for cid in stale:
                    del self._context_locks[cid]
                logger.debug("Cleaned up %d idle context locks", len(stale))

    # ------------------------------------------------------------------

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancel is not supported."""
        raise Exception("cancel not supported")


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------


def _create_task_store():
    """Create the appropriate TaskStore based on configuration.

    Uses A2A SDK's DatabaseTaskStore (PostgreSQL) when TASK_STORE_DB_URL
    is set. Falls back to InMemoryTaskStore for dev/test.

    This is A2A-generic — works for any agent framework, not just LangGraph.
    """
    import os

    db_url = os.environ.get("TASK_STORE_DB_URL", "")
    if db_url and _HAS_SQL_STORE:
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(
            db_url,
            pool_size=5,
            max_overflow=3,
            pool_recycle=300,  # Recycle connections every 5 min
            pool_pre_ping=True,  # Verify connection before use
            connect_args={"ssl": False},
        )
        store = DatabaseTaskStore(engine)
        logger.info("Using PostgreSQL TaskStore: %s", db_url.split("@")[-1])
        return store

    logger.info("Using InMemoryTaskStore (set TASK_STORE_DB_URL for persistence)")
    return InMemoryTaskStore()


def run() -> None:
    """Create the A2A server application and run it with uvicorn."""
    agent_card = get_agent_card(host="0.0.0.0", port=8000)

    request_handler = DefaultRequestHandler(
        agent_executor=SandboxAgentExecutor(),
        task_store=_create_task_store(),
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
