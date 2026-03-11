"""Budget tracking for the plan-execute-reflect reasoning loop.

Prevents runaway execution by capping iterations, tool calls per step,
total token usage, and wall clock time. When the budget is exceeded the
reflector forces the loop to terminate gracefully.

Token budget is enforced via LiteLLM as the single source of truth:
the agent queries the backend's ``/token-usage/sessions/{context_id}``
endpoint before each LLM call. This tracks ALL calls including
sub-agents (explore, delegate) and persists across restarts.

Budget scopes:
- **Per-message** (single graph run): max_iterations, max_wall_clock_s, recursion_limit
- **Per-step** (within one plan step): max_tool_calls_per_step
- **Per-session** (across A2A turns + restarts): token budget via LiteLLM

Budget parameters are configurable via environment variables:

- ``SANDBOX_MAX_ITERATIONS`` (default: 100)
- ``SANDBOX_MAX_TOOL_CALLS_PER_STEP`` (default: 10)
- ``SANDBOX_MAX_TOKENS`` (default: 1000000) — enforced via LiteLLM query
- ``SANDBOX_MAX_WALL_CLOCK_S`` (default: 600) — max seconds per message
- ``SANDBOX_HITL_INTERVAL`` (default: 50)
- ``SANDBOX_RECURSION_LIMIT`` (default: 50)
- ``SANDBOX_LLM_TIMEOUT`` (default: 300) — seconds per LLM call
- ``SANDBOX_LLM_MAX_RETRIES`` (default: 3) — retry on transient LLM errors
- ``KAGENTI_BACKEND_URL`` — backend URL for token-usage API
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

# Default backend URL for token-usage queries (in-cluster service discovery)
_DEFAULT_BACKEND_URL = os.environ.get(
    "KAGENTI_BACKEND_URL",
    "http://kagenti-backend.kagenti-system.svc.cluster.local:8000",
)

# Minimum seconds between LiteLLM usage queries (cache to avoid hammering)
_BUDGET_CHECK_INTERVAL = int(os.environ.get("SANDBOX_BUDGET_CHECK_INTERVAL", "5"))


def _env_int(name: str, default: int) -> int:
    """Read an integer from the environment, falling back to *default*."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class AgentBudget:
    """Tracks resource usage across the reasoning loop.

    Attributes
    ----------
    max_iterations:
        Maximum outer-loop iterations (planner → executor → reflector).
    max_tool_calls_per_step:
        Maximum tool invocations the executor may make for a single plan step.
    max_tokens:
        Approximate upper bound on total tokens consumed (prompt + completion).
    max_wall_clock_s:
        Maximum wall clock time in seconds for a single message run.
    hitl_interval:
        After this many iterations, the reflector suggests a human check-in.
    recursion_limit:
        LangGraph recursion limit passed to graph invocation config.
    """

    max_iterations: int = _env_int("SANDBOX_MAX_ITERATIONS", 100)
    max_tool_calls_per_step: int = _env_int("SANDBOX_MAX_TOOL_CALLS_PER_STEP", 10)
    max_tokens: int = _env_int("SANDBOX_MAX_TOKENS", 1_000_000)
    max_wall_clock_s: int = _env_int("SANDBOX_MAX_WALL_CLOCK_S", 600)
    hitl_interval: int = _env_int("SANDBOX_HITL_INTERVAL", 50)
    recursion_limit: int = _env_int("SANDBOX_RECURSION_LIMIT", 300)
    llm_timeout: int = _env_int("SANDBOX_LLM_TIMEOUT", 300)
    llm_max_retries: int = _env_int("SANDBOX_LLM_MAX_RETRIES", 3)

    # Mutable runtime counters — not constructor args.
    iterations_used: int = field(default=0, init=False)
    tokens_used: int = field(default=0, init=False)
    tool_calls_this_step: int = field(default=0, init=False)
    _start_time: float = field(default_factory=time.monotonic, init=False)
    _last_litellm_check: float = field(default=0.0, init=False)
    _session_id: str = field(default="", init=False)

    # -- helpers -------------------------------------------------------------

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for LiteLLM usage queries."""
        self._session_id = session_id

    def tick_iteration(self) -> None:
        """Advance the iteration counter by one."""
        self.iterations_used += 1

    def add_tokens(self, count: int) -> None:
        """Accumulate *count* tokens (prompt + completion).

        This is a fallback counter used when LiteLLM is unavailable.
        When LiteLLM is reachable, ``refresh_from_litellm`` overwrites
        ``tokens_used`` with the authoritative value.
        """
        self.tokens_used += count
        if self.tokens_exceeded:
            logger.warning(
                "Budget: tokens exceeded %d/%d",
                self.tokens_used, self.max_tokens,
            )

    async def refresh_from_litellm(self) -> None:
        """Query LiteLLM for actual session token usage.

        Updates ``tokens_used`` with the authoritative value from LiteLLM.
        Caches for ``_BUDGET_CHECK_INTERVAL`` seconds to avoid hammering.
        Falls back silently to the in-memory counter on error.
        """
        if not self._session_id:
            return

        now = time.monotonic()
        if now - self._last_litellm_check < _BUDGET_CHECK_INTERVAL:
            return  # Use cached value

        try:
            url = f"{_DEFAULT_BACKEND_URL}/api/v1/token-usage/sessions/{self._session_id}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    litellm_total = data.get("total_tokens", 0)
                    if litellm_total > 0:
                        self.tokens_used = litellm_total
                        self._last_litellm_check = now
                        logger.debug(
                            "Budget: LiteLLM reports %d tokens for session %s",
                            litellm_total, self._session_id[:12],
                        )
                else:
                    logger.debug(
                        "Budget: token-usage API returned %d for session %s",
                        resp.status_code, self._session_id[:12],
                    )
        except Exception as exc:
            logger.debug(
                "Budget: LiteLLM query failed for session %s: %s (using in-memory fallback)",
                self._session_id[:12], exc,
            )

    def tick_tool_call(self) -> None:
        """Record a tool invocation within the current step."""
        self.tool_calls_this_step += 1

    def reset_step_tools(self) -> None:
        """Reset the per-step tool-call counter (called between plan steps)."""
        self.tool_calls_this_step = 0

    # -- queries -------------------------------------------------------------

    @property
    def wall_clock_s(self) -> float:
        """Seconds elapsed since this budget was created."""
        return time.monotonic() - self._start_time

    @property
    def iterations_exceeded(self) -> bool:
        return self.iterations_used >= self.max_iterations

    @property
    def tokens_exceeded(self) -> bool:
        return self.tokens_used >= self.max_tokens

    @property
    def wall_clock_exceeded(self) -> bool:
        return self.wall_clock_s >= self.max_wall_clock_s

    @property
    def step_tools_exceeded(self) -> bool:
        return self.tool_calls_this_step >= self.max_tool_calls_per_step

    @property
    def exceeded(self) -> bool:
        """Return True if *any* budget limit has been reached."""
        return self.iterations_exceeded or self.tokens_exceeded or self.wall_clock_exceeded

    @property
    def exceeded_reason(self) -> str | None:
        """Human-readable reason for why the budget was exceeded, or None."""
        if self.iterations_exceeded:
            return f"Iteration limit reached ({self.iterations_used}/{self.max_iterations})"
        if self.tokens_exceeded:
            return f"Token limit reached ({self.tokens_used:,}/{self.max_tokens:,})"
        if self.wall_clock_exceeded:
            return f"Time limit reached ({self.wall_clock_s:.0f}s/{self.max_wall_clock_s}s)"
        return None

    @property
    def needs_hitl_checkin(self) -> bool:
        """Return True when it's time for a human-in-the-loop check-in."""
        return (
            self.hitl_interval > 0
            and self.iterations_used > 0
            and self.iterations_used % self.hitl_interval == 0
        )

    def summary(self) -> dict:
        """Return budget state as a dict for event serialization."""
        return {
            "tokens_used": self.tokens_used,
            "tokens_budget": self.max_tokens,
            "iterations_used": self.iterations_used,
            "iterations_budget": self.max_iterations,
            "wall_clock_s": round(self.wall_clock_s, 1),
            "max_wall_clock_s": self.max_wall_clock_s,
        }
