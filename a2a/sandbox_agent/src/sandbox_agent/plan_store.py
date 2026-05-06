"""Append-only nested plan container.

Stores the agent's execution plan as a nested structure of main steps
and subplans. Only additions are allowed after initial creation — the
replanner can add new main steps (after all existing are terminal) or
create alternative subplans within a step.

Structure::

    {
        "version": 1,
        "steps": {
            "1": {
                "description": "Clone the repo",
                "status": "done",
                "subplans": {
                    "a": {
                        "substeps": {
                            "1": {"description": "git clone ...", "status": "done"},
                        },
                        "status": "done",
                        "created_by": "planner",
                    }
                },
                "active_subplan": "a",
            },
            "2": {
                "description": "Analyze CI logs",
                "status": "running",
                "subplans": {
                    "a": {"substeps": {...}, "status": "failed", "created_by": "planner"},
                    "b": {"substeps": {...}, "status": "running", "created_by": "replanner"},
                },
                "active_subplan": "b",
            },
        },
    }

Status transitions (one-way):
    pending → running → done | failed | cancelled
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Valid status values and their terminal flag
_TERMINAL = frozenset({"done", "failed", "cancelled"})
_VALID_STATUS = frozenset({"pending", "running"}) | _TERMINAL


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def create_plan(steps: list[str], creator: str = "planner") -> dict[str, Any]:
    """Create a new plan store from a list of step descriptions.

    Each step gets a single subplan "a" with one substep matching
    the step description (for simple plans where steps = substeps).
    """
    plan: dict[str, Any] = {"version": 1, "steps": {}}
    for i, desc in enumerate(steps):
        step_key = str(i + 1)
        plan["steps"][step_key] = {
            "description": desc,
            "status": "pending",
            "subplans": {
                "a": {
                    "substeps": {
                        "1": {"description": desc, "status": "pending"},
                    },
                    "status": "pending",
                    "created_by": creator,
                },
            },
            "active_subplan": "a",
        }
    # Mark first step as running
    if plan["steps"]:
        plan["steps"]["1"]["status"] = "running"
        plan["steps"]["1"]["subplans"]["a"]["status"] = "running"
    return plan


# ---------------------------------------------------------------------------
# Mutations (append-only)
# ---------------------------------------------------------------------------


def add_steps(
    plan: dict[str, Any],
    new_steps: list[str],
    creator: str = "replanner",
) -> dict[str, Any]:
    """Add new main steps to the plan.

    Only allowed when ALL existing steps are terminal (done/failed/cancelled).
    Returns a new plan dict (does not mutate in place).

    Raises ValueError if preconditions are not met.
    """
    if creator != "replanner":
        raise ValueError(f"Only replanner can add steps, got creator={creator}")

    steps = plan.get("steps", {})
    non_terminal = [k for k, s in steps.items() if s.get("status") not in _TERMINAL]
    if non_terminal:
        raise ValueError(f"Cannot add steps: steps {non_terminal} are still active")

    new_plan = _deep_copy(plan)
    next_idx = max((int(k) for k in steps), default=0) + 1
    for i, desc in enumerate(new_steps):
        step_key = str(next_idx + i)
        new_plan["steps"][step_key] = {
            "description": desc,
            "status": "pending",
            "subplans": {
                "a": {
                    "substeps": {
                        "1": {"description": desc, "status": "pending"},
                    },
                    "status": "pending",
                    "created_by": creator,
                },
            },
            "active_subplan": "a",
        }

    # Mark first new step as running
    first_new = str(next_idx)
    if first_new in new_plan["steps"]:
        new_plan["steps"][first_new]["status"] = "running"
        new_plan["steps"][first_new]["subplans"]["a"]["status"] = "running"

    logger.info(
        "Added %d steps (start=%s) by %s",
        len(new_steps),
        first_new,
        creator,
    )
    return new_plan


def add_alternative_subplan(
    plan: dict[str, Any],
    step_key: str,
    substeps: list[str],
) -> tuple[dict[str, Any], str]:
    """Create an alternative subplan for a step (replanner only).

    Returns (new_plan, subplan_key) where subplan_key is the new key (b, c, ...).
    The active_subplan is switched to the new one.
    """
    new_plan = _deep_copy(plan)
    step = new_plan["steps"].get(step_key)
    if step is None:
        raise ValueError(f"Step {step_key} not found")

    existing_keys = sorted(step["subplans"].keys())
    next_key = chr(ord("a") + len(existing_keys))

    step["subplans"][next_key] = {
        "substeps": {str(i + 1): {"description": desc, "status": "pending"} for i, desc in enumerate(substeps)},
        "status": "running",
        "created_by": "replanner",
    }
    step["active_subplan"] = next_key
    step["status"] = "running"

    logger.info(
        "Created alternative subplan '%s' for step %s (%d substeps)",
        next_key,
        step_key,
        len(substeps),
    )
    return new_plan, next_key


# ---------------------------------------------------------------------------
# Status updates
# ---------------------------------------------------------------------------


def set_step_status(
    plan: dict[str, Any],
    step_key: str,
    status: str,
) -> dict[str, Any]:
    """Update a step's status. Validates one-way transitions."""
    if status not in _VALID_STATUS:
        raise ValueError(f"Invalid status: {status}")
    new_plan = _deep_copy(plan)
    step = new_plan["steps"].get(step_key)
    if step is None:
        raise ValueError(f"Step {step_key} not found")
    old = step["status"]
    if old in _TERMINAL:
        logger.warning("Step %s already terminal (%s), ignoring → %s", step_key, old, status)
        return new_plan
    step["status"] = status
    # Also update the active subplan status
    active = step.get("active_subplan", "a")
    if active in step.get("subplans", {}):
        sp = step["subplans"][active]
        if sp.get("status") not in _TERMINAL:
            sp["status"] = status
    return new_plan


def set_substep_status(
    plan: dict[str, Any],
    step_key: str,
    substep_key: str,
    status: str,
    result_summary: str = "",
    tool_calls: list[str] | None = None,
) -> dict[str, Any]:
    """Update a substep's status within the active subplan."""
    if status not in _VALID_STATUS:
        raise ValueError(f"Invalid status: {status}")
    new_plan = _deep_copy(plan)
    step = new_plan["steps"].get(step_key)
    if step is None:
        raise ValueError(f"Step {step_key} not found")
    active = step.get("active_subplan", "a")
    subplan = step.get("subplans", {}).get(active)
    if subplan is None:
        raise ValueError(f"Subplan {active} not found in step {step_key}")
    substep = subplan.get("substeps", {}).get(substep_key)
    if substep is None:
        raise ValueError(f"Substep {substep_key} not found in subplan {active}")
    substep["status"] = status
    if result_summary:
        substep["result_summary"] = result_summary
    if tool_calls:
        substep["tool_calls"] = tool_calls
    return new_plan


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def get_current_step(plan: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Return (step_key, step_dict) for the first non-terminal step."""
    for key in sorted(plan.get("steps", {}), key=int):
        step = plan["steps"][key]
        if step.get("status") not in _TERMINAL:
            return key, step
    return None


def get_active_substep(plan: dict[str, Any], step_key: str) -> tuple[str, dict] | None:
    """Return (substep_key, substep_dict) for the first pending/running substep."""
    step = plan.get("steps", {}).get(step_key)
    if step is None:
        return None
    active = step.get("active_subplan", "a")
    subplan = step.get("subplans", {}).get(active)
    if subplan is None:
        return None
    for sk in sorted(subplan.get("substeps", {}), key=int):
        ss = subplan["substeps"][sk]
        if ss.get("status") not in _TERMINAL:
            return sk, ss
    return None


def step_count(plan: dict[str, Any]) -> int:
    """Total number of main steps."""
    return len(plan.get("steps", {}))


def done_count(plan: dict[str, Any]) -> int:
    """Number of completed main steps."""
    return sum(1 for s in plan.get("steps", {}).values() if s.get("status") == "done")


def all_terminal(plan: dict[str, Any]) -> bool:
    """True if ALL main steps are in a terminal status."""
    steps = plan.get("steps", {})
    return bool(steps) and all(s.get("status") in _TERMINAL for s in steps.values())


def to_flat_plan(plan: dict[str, Any]) -> list[str]:
    """Convert to flat list of step descriptions (backward compat)."""
    return [plan["steps"][k]["description"] for k in sorted(plan.get("steps", {}), key=int)]


def to_flat_plan_steps(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert to flat PlanStep list (backward compat with serializer/UI)."""
    result = []
    for key in sorted(plan.get("steps", {}), key=int):
        step = plan["steps"][key]
        active = step.get("active_subplan", "a")
        subplan = step.get("subplans", {}).get(active, {})
        alt_count = len(step.get("subplans", {})) - 1  # alternatives (excl. original)
        result.append(
            {
                "index": int(key) - 1,  # 0-based for compat
                "description": step["description"],
                "status": step["status"],
                "active_subplan": active,
                "alternative_count": alt_count,
                "substeps": list(subplan.get("substeps", {}).values()),
                "created_by": subplan.get("created_by", "planner"),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _deep_copy(d: dict) -> dict:
    """Fast deep copy for JSON-compatible dicts."""
    import json

    return json.loads(json.dumps(d))
