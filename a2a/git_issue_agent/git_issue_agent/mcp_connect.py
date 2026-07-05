"""Reusable helper for connecting to an MCP server without hanging on failure.

Background
----------
``crewai_tools.MCPServerAdapter`` connects via ``mcpadapt``. Under the hood
``mcpadapt`` runs the connection inside a *daemon thread* and the synchronous
entry point blocks the caller on a single ``threading.Event`` wait:

    if not self.ready.wait(timeout=connect_timeout):
        raise TimeoutError(...)

If the connection *fails* for any reason (auth refused, HTTP 401, connection
refused, DNS error, ...), the exception is raised inside the daemon thread and
is **never re-raised on the calling thread**. ``ready`` is simply never set, so
the caller blocks for the full ``connect_timeout`` (600s in this agent) before
getting a generic ``TimeoutError``. Because this happens synchronously inside an
``async def`` handler, it also freezes the event loop and starves other tasks.

This module fixes that by owning the wait itself. It starts the connection and
then polls two signals on the underlying ``MCPAdapt`` object:

* ``ready`` (a ``threading.Event``) -- set once the connection succeeds.
* ``thread`` (a daemon ``threading.Thread``) -- dies when the connect coroutine
  raises.

That yields three clean states:

============================ ============= ========== ====================
state                        thread alive? ready set? action
============================ ============= ========== ====================
connecting / OAuth login     yes           no         keep waiting
connect failed               **no**        no         **fail immediately**
connected                    yes           yes        proceed with tools
============================ ============= ========== ====================

So a user taking minutes to complete an interactive OAuth login is still
supported (thread stays alive -> we wait, up to ``connect_timeout``), while a
hard failure aborts within one poll interval instead of after ``connect_timeout``.

The helper is written generically -- there is nothing GitHub-specific here -- so
any agent in this repo can copy it and use ``mcp_tools_session``.

Version note
------------
The fast-fail path pokes at ``mcpadapt`` internals (``MCPAdapt.ready`` /
``MCPAdapt.thread`` / ``MCPAdapt.tools()``), verified against ``crewai-tools``
1.6.1 (mcpadapt bundled therein). All internal access is guarded with
``getattr``; if the attributes are missing (library changed), we fall back to
the original blocking ``MCPServerAdapter`` behavior so we never crash -- we only
lose the fast-fail optimization, with a warning logged.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from crewai_tools import MCPServerAdapter
from crewai_tools.adapters.tool_collection import ToolCollection

logger = logging.getLogger(__name__)


class McpConnectError(RuntimeError):
    """Raised when an MCP connection fails or times out.

    The real underlying cause is chained via ``raise ... from`` when available.
    """


@asynccontextmanager
async def mcp_tools_session(
    server_params: dict,
    *,
    connect_timeout: int,
    poll_interval: float = 1.0,
):
    """Connect to an MCP server, yielding its tools for the duration of the block.

    Fails fast on connection errors while capping a genuine never-returns hang at
    ``connect_timeout`` seconds. The MCP session stays open for the body of the
    ``async with`` and is torn down on exit (success or failure).

    Args:
        server_params: Passed straight to ``MCPServerAdapter`` (e.g. ``url`` /
            ``transport`` / ``headers``).
        connect_timeout: Last-resort ceiling, in seconds, for a connection that
            neither succeeds nor fails (e.g. a stuck interactive OAuth flow).
        poll_interval: How often, in seconds, to check the connection state.
            Small values surface failures almost instantly at negligible cost.

    Yields:
        A ``ToolCollection`` of the adapted MCP tools.

    Raises:
        McpConnectError: on connection failure or on hitting ``connect_timeout``.
    """
    adapter = _build_adapter(server_params, connect_timeout)

    # Fallback: internals not as expected -> use the library's own blocking
    # context manager. Correct, just without the fast-fail optimization.
    if adapter is None:
        logger.warning(
            "mcpadapt internals not recognized; falling back to blocking connect "
            "(a failed connection may take up to %ss to surface)",
            connect_timeout,
        )
        blocking = MCPServerAdapter(server_params, connect_timeout=connect_timeout)
        try:
            with blocking as tools:
                yield tools
        finally:
            pass  # `with` handles teardown
        return

    server_adapter, mcp_adapter = adapter
    try:
        tools = await _connect_with_fast_fail(mcp_adapter, connect_timeout, poll_interval)
        # Wrap the raw adapted tools the same way MCPServerAdapter.tools does.
        yield ToolCollection(tools)
    finally:
        # Cancel the keep-alive task, join the daemon thread, close the loop.
        try:
            server_adapter.stop()
        except Exception as exc:  # noqa: BLE001 - teardown must not mask the real error
            logger.warning("Error during MCP adapter teardown: %s", exc)


def _build_adapter(server_params: dict, connect_timeout: int):
    """Construct an MCPServerAdapter and start its connection WITHOUT blocking.

    Returns ``(server_adapter, mcp_adapter)`` on success, or ``None`` if the
    library internals we depend on are not present (caller should fall back).

    We replicate ``mcpadapt``'s own start sequence -- ``thread.start()`` -- rather
    than calling the library's ``start()``, which would block on ``ready.wait``.
    """
    try:
        # Build the object graph without triggering the blocking start(). We
        # can't call MCPServerAdapter(...) directly because its __init__ calls
        # start(); instead build the underlying MCPAdapt ourselves, mirroring
        # crewai_tools' construction, and attach it to a bare MCPServerAdapter.
        from mcpadapt.core import MCPAdapt
        from mcpadapt.crewai_adapter import CrewAIAdapter

        server_adapter = MCPServerAdapter.__new__(MCPServerAdapter)
        server_adapter._adapter = None
        server_adapter._tools = None
        server_adapter._tool_names = None
        server_adapter._serverparams = server_params

        mcp_adapter = MCPAdapt(server_params, CrewAIAdapter(), connect_timeout)
        server_adapter._adapter = mcp_adapter

        # Sanity-check the internals we will poll before committing to this path.
        thread = getattr(mcp_adapter, "thread", None)
        ready = getattr(mcp_adapter, "ready", None)
        if thread is None or ready is None or not hasattr(mcp_adapter, "tools"):
            return None

        thread.start()  # kick off the connection; do NOT call start()/ __enter__
        return server_adapter, mcp_adapter
    except Exception as exc:  # noqa: BLE001 - any surprise -> fall back to blocking
        logger.warning("Could not set up fast-fail MCP connect (%s); will fall back", exc)
        return None


async def _connect_with_fast_fail(mcp_adapter, connect_timeout: int, poll_interval: float):
    """Poll the connection until ready, dead, or timed out.

    Returns the adapted tool list on success. Raises ``McpConnectError`` on a
    dead connect thread (fast path) or on exceeding ``connect_timeout``.
    """
    ready = mcp_adapter.ready
    thread = mcp_adapter.thread

    waited = 0.0
    while True:
        if ready.is_set():
            break
        if not thread.is_alive():
            # The connect coroutine raised and killed its thread. mcpadapt
            # discards the exception, so we surface a clear error of our own.
            raise McpConnectError(
                "MCP connection failed (the connection attempt terminated before "
                "becoming ready). Check the MCP URL, network reachability, and auth."
            )
        if waited >= connect_timeout:
            raise McpConnectError(f"MCP connection did not become ready within {connect_timeout}s.")
        await asyncio.sleep(poll_interval)
        waited += poll_interval

    # Connection is ready; produce the adapted tools. tools() drives the loop in
    # the worker thread, so failures here are real and should also surface fast.
    try:
        return await asyncio.to_thread(mcp_adapter.tools)
    except Exception as exc:  # noqa: BLE001 - surface the real cause
        raise McpConnectError(f"Failed to list MCP tools: {exc}") from exc
