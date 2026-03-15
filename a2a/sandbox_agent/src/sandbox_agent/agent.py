"""A2A agent server for the Sandbox Legion.

Wires together the workspace manager, permission checker, sources config,
and LangGraph graph to serve the A2A protocol over HTTP.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from textwrap import dedent
from typing import Any

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
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentSkill,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message, new_task
from langchain_core.messages import HumanMessage
from starlette.routing import Route

from langgraph.checkpoint.memory import MemorySaver

from sandbox_agent.budget import AgentBudget
from sandbox_agent.configuration import Configuration
from sandbox_agent.event_serializer import LangGraphSerializer
from sandbox_agent.graph import _load_skill, build_graph
from sandbox_agent.graph_card import build_graph_card
from sandbox_agent.observability import setup_observability, create_tracing_middleware
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
    # Write to /tmp to avoid PermissionError when OCP assigns arbitrary UID
    # (the /app directory is owned by UID 1001 but OCP may run as a different UID)
    hash_file = Path("/tmp") / _TOFU_HASH_FILE
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
    capabilities = AgentCapabilities(
        streaming=True,
        extensions=[
            AgentExtension(
                uri="urn:kagenti:agent-graph-card:v1",
                description="Processing graph topology and event schemas",
                required=False,
                params={"endpoint": "/.well-known/agent-graph-card.json"},
            ),
        ],
    )
    # Scan workspace for loaded skill files (.claude/skills/**/*.md)
    # Skills found on disk are advertised in the agent card so the UI
    # can show them in the / autocomplete (SkillWhisperer).
    skills: list[AgentSkill] = []
    workspace = os.environ.get("WORKSPACE_DIR", "/workspace")
    skills_dir = Path(workspace) / ".claude" / "skills"
    if skills_dir.is_dir():
        seen_ids: set[str] = set()
        for md_file in sorted(skills_dir.rglob("SKILL.md")):
            # Directory-based skills: auth:keycloak-confidential-client/SKILL.md
            # Skill ID = directory name relative to skills_dir
            rel_dir = md_file.parent.relative_to(skills_dir)
            skill_id = str(rel_dir).replace("/", ":")
            if skill_id in seen_ids or skill_id == ".":
                continue
            seen_ids.add(skill_id)
            # Read description from the skill file (skip frontmatter)
            try:
                content = md_file.read_text(errors="replace")
                desc = ""
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("description:"):
                        desc = line.split(":", 1)[1].strip().strip("'\"")
                        break
                    if line.startswith("# ") and not desc:
                        desc = line.lstrip("# ").strip()
                if not desc:
                    desc = skill_id
            except Exception:
                desc = skill_id
            skills.append(
                AgentSkill(
                    id=skill_id,
                    name=skill_id,
                    description=desc[:200],
                    tags=["skill"],
                )
            )
        logger.info("Found %d skills in %s", len(skills), skills_dir)

    # Always include the base sandbox skill
    skills.append(
        AgentSkill(
            id="sandbox_legion",
            name="Sandbox Legion",
            description=(
                "Sandboxed coding assistant with shell execution, file read/write, "
                "web fetch, explore, and delegate capabilities."
            ),
            tags=["shell", "file", "workspace", "sandbox"],
            examples=[
                "Run 'ls -la' in my workspace",
                "Create a Python script that prints hello world",
                "Read the contents of output/results.txt",
            ],
        )
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
        skills=skills,
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
        namespace = os.environ.get("NAMESPACE", "team1")
        graph = build_graph(
            workspace_path=workspace_path,
            permission_checker=self._permission_checker,
            sources_config=self._sources_config,
            checkpointer=self._checkpointer,
            context_id=context_id or "stateless",
            namespace=namespace,
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
            input_state: dict[str, Any] = {
                "messages": messages,
                "workspace_path": workspace_path,
                "context_id": context_id or "stateless",
            }

            # Extract skill from A2A message metadata and load its content.
            # TODO(Session N): Once base image moves to kagenti repo, use
            # skill_pack_loader.py at startup to clone verified skill packs
            # from skill-packs.yaml into /workspace/.claude/skills/ before
            # the first message. Currently skills must be pre-populated.
            msg = context.message
            skill_id = None
            if msg and msg.metadata:
                skill_id = msg.metadata.get("skill")

            if skill_id:
                skill_content = _load_skill(workspace_path, skill_id)
                if skill_content:
                    input_state["skill_instructions"] = (
                        f'<skill name="{skill_id}">\n'
                        f"{skill_content}\n"
                        f"</skill>\n\n"
                        f"Follow the skill instructions above for this task."
                    )
                    logger.info("Loaded skill '%s' for context_id=%s", skill_id, context_id)
                else:
                    logger.warning("Skill '%s' requested but not found in workspace %s", skill_id, workspace_path)

            graph_config = {
                "configurable": {"thread_id": context_id or "stateless"},
                "recursion_limit": AgentBudget().recursion_limit,
            }
            logger.info("Processing messages: %s (thread_id=%s)", input_state, context_id)

            try:
                output = None
                serializer = LangGraphSerializer(context_id=context_id)
                llm_request_ids: list[str] = []

                # Run graph in a shielded background task so client disconnect
                # does NOT cancel the LangGraph execution.  Events are fed
                # through an asyncio.Queue; the consumer (below) forwards them
                # to the A2A event stream.  If the consumer is cancelled the
                # graph keeps running and saves results to the task store.
                _SENTINEL = object()
                event_queue: asyncio.Queue = asyncio.Queue()

                async def _run_graph() -> None:
                    """Execute graph and push events to queue (shielded)."""
                    max_retries = 3
                    for attempt in range(max_retries + 1):
                        try:
                            async for ev in graph.astream(
                                input_state, config=graph_config, stream_mode="updates"
                            ):
                                await event_queue.put(ev)
                            break  # success
                        except Exception as retry_err:
                            err_str = str(retry_err).lower()
                            is_quota = "insufficient_quota" in err_str
                            is_rate = "rate_limit" in err_str or "429" in err_str
                            if is_quota:
                                logger.error("LLM quota exceeded: %s", retry_err)
                                await event_queue.put(
                                    {"_error": "LLM API quota exceeded. Check billing."}
                                )
                                break
                            elif is_rate and attempt < max_retries:
                                delay = 2 ** (attempt + 1)
                                logger.warning(
                                    "Rate limited (%d/%d), retrying in %ds: %s",
                                    attempt + 1, max_retries, delay, retry_err,
                                )
                                await asyncio.sleep(delay)
                                continue
                            else:
                                logger.error("Graph execution failed: %s", retry_err, exc_info=True)
                                await event_queue.put({"_error": str(retry_err)})
                                break
                    await event_queue.put(_SENTINEL)

                # Shield the graph task from cancellation
                graph_task = asyncio.ensure_future(asyncio.shield(_run_graph()))

                # Consume events from the queue — this side CAN be cancelled
                event_count = 0
                client_disconnected = False
                while True:
                    try:
                        event = await event_queue.get()
                    except asyncio.CancelledError:
                        logger.warning(
                            "Event consumer cancelled (context=%s) — graph continues in background",
                            context_id,
                        )
                        client_disconnected = True
                        break
                    if event is _SENTINEL:
                        break
                    if "_error" in event:
                        error_msg = event["_error"]
                        await task_updater.update_status(
                            TaskState.working,
                            new_agent_text_message(
                                json.dumps({"type": "error", "message": error_msg}),
                                task_updater.context_id,
                                task_updater.task_id,
                            ),
                        )
                        parts = [TextPart(text=f"Error: {error_msg}")]
                        await task_updater.add_artifact(parts)
                        await task_updater.failed()
                        return

                    event_count += 1
                    node_names = list(event.keys())
                    logger.info(
                        "Graph event %d: nodes=%s (context=%s)",
                        event_count, node_names, context_id,
                    )

                    # Skip __interrupt__ events (HITL pause) — these contain
                    # tuples, not dicts, and shouldn't be serialized.
                    if "__interrupt__" in event:
                        logger.info(
                            "Graph interrupted (HITL) at event %d: %s",
                            event_count, event.get("__interrupt__"),
                        )
                        # Emit a structured HITL event for the frontend
                        hitl_data = event.get("__interrupt__", ())
                        hitl_msg = str(hitl_data[0]) if hitl_data else "Approval required"
                        hitl_json = json.dumps({
                            "type": "hitl_request",
                            "loop_id": serializer._loop_id,
                            "message": hitl_msg[:500],
                        })
                        await task_updater.update_status(
                            TaskState.working,
                            new_agent_text_message(
                                hitl_json + "\n",
                                task_updater.context_id,
                                task_updater.task_id,
                            ),
                        )
                        continue

                    # Send intermediate status updates as structured JSON
                    try:
                        serialized_lines = "\n".join(
                            serializer.serialize(key, value)
                            for key, value in event.items()
                            if isinstance(value, dict)
                        ) + "\n"
                        await task_updater.update_status(
                            TaskState.working,
                            new_agent_text_message(
                                serialized_lines,
                                task_updater.context_id,
                                task_updater.task_id,
                            ),
                        )
                        line_types = []
                        for line in serialized_lines.split("\n"):
                            line = line.strip()
                            if line:
                                try:
                                    lt = json.loads(line).get("type", "?")
                                    line_types.append(lt)
                                except json.JSONDecodeError:
                                    line_types.append("parse_error")
                        logger.info("A2A_EMIT session=%s lines=%d types=%s",
                            context_id, len(line_types), line_types)
                    except asyncio.CancelledError:
                        logger.warning(
                            "SSE update cancelled at event %d (context=%s) — client disconnected",
                            event_count, context_id,
                        )
                        client_disconnected = True
                        break
                    except Exception as update_err:
                        logger.error(
                            "Failed to send SSE update for event %d: %s",
                            event_count, update_err,
                        )
                    output = event

                    # Capture LLM request_ids from AIMessage responses
                    for _node_val in event.values():
                        if isinstance(_node_val, dict):
                            for _msg in _node_val.get("messages", []):
                                _rid = getattr(_msg, "response_metadata", {}).get("id")
                                if _rid and _rid not in llm_request_ids:
                                    llm_request_ids.append(_rid)

                # If client disconnected, wait for graph to finish in background
                if client_disconnected:
                    logger.info("Waiting for graph to complete in background (context=%s)", context_id)
                    try:
                        await asyncio.wait_for(graph_task, timeout=300)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        logger.warning("Graph background task timed out or cancelled (context=%s)", context_id)
                    # Drain remaining events — serialize and persist them
                    # since the SSE consumer was cancelled and missed these.
                    bg_event_count = 0
                    bg_serialized_lines: list[str] = []
                    while not event_queue.empty():
                        ev = event_queue.get_nowait()
                        if ev is _SENTINEL or "_error" in ev:
                            continue
                        output = ev
                        bg_event_count += 1
                        # Serialize each event so it can be persisted
                        try:
                            for key, value in ev.items():
                                if isinstance(value, dict):
                                    serialized = serializer.serialize(key, value)
                                    bg_serialized_lines.append(serialized)
                        except Exception as ser_err:
                            logger.warning("Failed to serialize bg event %d: %s", bg_event_count, ser_err)
                    if bg_event_count > 0:
                        logger.info(
                            "Drained %d background events for context=%s, serialized %d lines",
                            bg_event_count, context_id, len(bg_serialized_lines),
                        )
                        # Persist via task_updater so the events appear in history
                        for line_block in bg_serialized_lines:
                            try:
                                await task_updater.update_status(
                                    TaskState.working,
                                    new_agent_text_message(
                                        line_block + "\n",
                                        task_updater.context_id,
                                        task_updater.task_id,
                                    ),
                                )
                            except Exception:
                                pass  # best-effort

                # Extract final answer from the last event.
                # The reporter node sets {"final_answer": "..."}.
                # Fall back to checking messages from reporter or executor.
                final_answer = None
                if output:
                    # 1. Check reporter node output (plan-execute-reflect)
                    reporter_output = output.get("reporter", {})
                    if isinstance(reporter_output, dict):
                        final_answer = reporter_output.get("final_answer")

                    # 2. Fall back to executor/assistant message content
                    if not final_answer:
                        for node_name in ("reporter", "executor", "assistant"):
                            node_output = output.get(node_name, {})
                            if isinstance(node_output, dict):
                                msgs = node_output.get("messages", [])
                                if msgs:
                                    content = getattr(msgs[-1], "content", None)
                                    if isinstance(content, list):
                                        final_answer = "\n".join(
                                            block.get("text", "") if isinstance(block, dict) else str(block)
                                            for block in content
                                            if isinstance(block, dict) and block.get("type") == "text"
                                        ) or None
                                    elif content:
                                        final_answer = str(content)
                                    if final_answer:
                                        break

                if final_answer is None:
                    final_answer = "No response generated."

                # Store LLM request_ids in task metadata for token usage tracking
                if llm_request_ids:
                    try:
                        existing_meta = {}
                        if task.metadata:
                            existing_meta = dict(task.metadata) if not isinstance(task.metadata, dict) else task.metadata
                        existing_meta["llm_request_ids"] = llm_request_ids
                        task.metadata = existing_meta
                        logger.info(
                            "Stored %d LLM request_ids in task metadata for context_id=%s",
                            len(llm_request_ids), context_id,
                        )
                    except Exception as meta_err:
                        logger.warning("Failed to store llm_request_ids: %s", meta_err)

                # Add artifact with final answer and complete
                parts = [TextPart(text=final_answer)]
                await task_updater.add_artifact(parts)
                await task_updater.complete()

            except asyncio.CancelledError:
                logger.warning(
                    "Graph execution context cancelled for context=%s — client likely disconnected. "
                    "Agent will continue processing and save results to task store.",
                    context_id,
                )
                # Don't return — fall through to save results to task store.
                # The A2A SDK persists the task, so the client can poll later.
            except Exception as e:
                logger.error("Graph execution error: %s", e, exc_info=True)
                error_msg = json.dumps({"type": "error", "message": str(e)})
                await task_updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        error_msg,
                        task_updater.context_id,
                        task_updater.task_id,
                    ),
                )
                parts = [TextPart(text=f"Error: {e}")]
                await task_updater.add_artifact(parts)
                await task_updater.failed()

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


class _MergingDatabaseTaskStore(DatabaseTaskStore):
    """DatabaseTaskStore that preserves backend-managed metadata fields.

    The backend writes fields like ``owner``, ``agent_name``, ``loop_events``
    to the ``metadata`` column. The default ``save()`` uses SQLAlchemy
    ``merge()`` which overwrites the entire row, losing those fields.

    This subclass reads existing metadata before writing and merges
    backend-managed keys so they survive A2A SDK updates.
    """

    _BACKEND_KEYS = frozenset({
        "owner", "visibility", "title", "agent_name", "loop_events",
    })

    async def save(self, task, context=None):
        """Save task while preserving backend-managed metadata fields."""
        await self._ensure_initialized()

        # Read existing metadata before overwriting
        existing_meta = {}
        async with self.async_session_maker() as session:
            from sqlalchemy import select
            stmt = select(self.task_model).where(self.task_model.id == task.id)
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            if existing and existing.task_metadata:
                raw = existing.task_metadata
                if isinstance(raw, dict):
                    existing_meta = raw
                elif isinstance(raw, str):
                    import json
                    try:
                        existing_meta = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        pass

        # Merge: start with new task metadata, overlay backend fields from existing
        merged = dict(task.metadata or {}) if task.metadata else {}
        for key in self._BACKEND_KEYS:
            if key in existing_meta and key not in merged:
                merged[key] = existing_meta[key]

        # Update task metadata with merged result
        task.metadata = merged if merged else task.metadata

        # Call parent save (which does session.merge)
        db_task = self._to_orm(task)
        async with self.async_session_maker.begin() as session:
            await session.merge(db_task)
            logger.debug("Task %s saved with merged metadata (keys=%s)",
                         task.id, list(merged.keys()) if merged else [])


def _create_task_store():
    """Create the appropriate TaskStore based on configuration.

    Uses _MergingDatabaseTaskStore (PostgreSQL) when TASK_STORE_DB_URL
    is set. Falls back to InMemoryTaskStore for dev/test.

    The merging store preserves backend-managed metadata fields (owner,
    agent_name, loop_events) that would otherwise be overwritten by
    the A2A SDK's session.merge().
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
        )
        store = _MergingDatabaseTaskStore(engine)
        logger.info("Using MergingDatabaseTaskStore: %s", db_url.split("@")[-1])
        return store

    logger.info("Using InMemoryTaskStore (set TASK_STORE_DB_URL for persistence)")
    return InMemoryTaskStore()


def _load_skill_packs_at_startup() -> None:
    """Clone skill repos into /workspace/.claude/skills/ at startup.

    Reads SKILL_REPOS env var (comma-separated git URLs with optional
    path suffix after #). Falls back to kagenti repo skills.

    TODO(Session N): Replace with skill_pack_loader.py once the base
    image moves to the kagenti repo.
    """
    import subprocess

    workspace = os.environ.get("WORKSPACE_DIR", "/workspace")
    skills_dir = Path(workspace) / ".claude" / "skills"

    if skills_dir.exists() and any(skills_dir.rglob("*.md")):
        logger.info("Skills already loaded at %s, skipping clone", skills_dir)
        return

    # Default: clone kagenti skills from the upstream public repo
    repos = os.environ.get(
        "SKILL_REPOS",
        "https://github.com/kagenti/kagenti.git#.claude/skills",
    )

    for entry in repos.split(","):
        entry = entry.strip()
        if not entry:
            continue

        # Parse "url@branch#path" format
        branch = None
        if "#" in entry:
            url_part, skill_path = entry.rsplit("#", 1)
        else:
            url_part, skill_path = entry, ".claude/skills"
        if "@" in url_part and not url_part.startswith("git@"):
            repo_url, branch = url_part.rsplit("@", 1)
        else:
            repo_url = url_part

        clone_dir = Path(workspace) / ".skill-repos" / repo_url.split("/")[-1].replace(".git", "")

        # Remove stale clone if exists (pod restart)
        if clone_dir.exists():
            subprocess.run(["rm", "-rf", str(clone_dir)], capture_output=True, timeout=10)

        try:
            cmd = ["git", "clone", "--depth", "1", "--single-branch"]
            if branch:
                cmd += ["--branch", branch]
            cmd += [repo_url, str(clone_dir)]
            logger.info("Cloning skills from %s branch=%s (path: %s)", repo_url, branch or "default", skill_path)
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            src = clone_dir / skill_path
            if src.is_dir():
                skills_dir.mkdir(parents=True, exist_ok=True)
                # Copy skill files (preserve directory structure)
                subprocess.run(
                    ["cp", "-r"] + [str(p) for p in src.iterdir()] + [str(skills_dir)],
                    capture_output=True,
                    timeout=30,
                )
                count = len(list(skills_dir.rglob("*.md")))
                logger.info("Loaded %d skill files from %s", count, repo_url)
            else:
                logger.warning("Skill path %s not found in %s", skill_path, repo_url)
        except subprocess.TimeoutExpired:
            logger.warning("Timeout cloning %s", repo_url)
        except Exception as e:
            logger.warning("Failed to clone skills from %s: %s", repo_url, e)


def run() -> None:
    """Create the A2A server application and run it with uvicorn."""
    # Initialize OTel GenAI auto-instrumentation (if OTEL_EXPORTER_OTLP_ENDPOINT is set)
    tracing_enabled = setup_observability()

    # Load skills from git repos before building the agent card
    _load_skill_packs_at_startup()

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

    # Add OTel tracing middleware (root span for every agent invocation)
    if tracing_enabled:
        from starlette.middleware.base import BaseHTTPMiddleware
        app.add_middleware(BaseHTTPMiddleware, dispatch=create_tracing_middleware())
        logger.info("OTel GenAI tracing middleware enabled")

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

    # Build the graph card from the compiled LangGraph.
    # We compile a temporary graph just for introspection (no checkpointer needed).
    _graph_card_cache: dict[str, Any] = {}

    async def _handle_graph_card(request: Any) -> Any:  # noqa: ARG001
        from starlette.responses import JSONResponse

        if not _graph_card_cache:
            # Build a graph for introspection only (no checkpointer, dummy config)
            from sandbox_agent.permissions import PermissionChecker
            from sandbox_agent.sources import SourcesConfig
            pc = PermissionChecker(settings={"workspace": "/workspace", "permissions": {}})
            sc = SourcesConfig()
            compiled = build_graph(
                workspace_path="/workspace",
                permission_checker=pc,
                sources_config=sc,
                checkpointer=None,
            )
            _graph_card_cache.update(
                build_graph_card(compiled, agent_id="sandbox-legion-v1")
            )
        return JSONResponse(_graph_card_cache)

    app.routes.insert(
        0,
        Route(
            "/.well-known/agent-graph-card.json",
            _handle_graph_card,
            methods=["GET"],
            name="agent_graph_card",
        ),
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)
