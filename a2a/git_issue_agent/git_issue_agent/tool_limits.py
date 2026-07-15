"""Bound the size of MCP tool output before it reaches the LLM.

Why
---
A broad issue query can return a very large payload (e.g. 81 issues for a "top 5"
request). Feeding that raw into the prompt overflows the model context window,
which then triggers CrewAI's expensive summarize-and-retry fallback. The reliable
fix is to cap what a tool returns *before* it is appended to the prompt, since a
prompt instruction alone cannot guarantee the model asks for less.

This module wraps a CrewAI ``BaseTool`` so that, transparently to CrewAI:

* an explicit fetch bound (``per_page``) is injected into the tool arguments when
  the tool's schema supports it and the caller did not already ask for less; and
* the returned observation is sliced to at most ``max_items`` list entries and
  then hard-capped to ``max_chars`` characters.

It is written generically -- it keys off the common ``{total_count, items:[...]}``
/ ``results`` MCP result shape and falls back to plain character truncation for
anything else -- so any agent can reuse it. There is nothing GitHub-specific here.
"""

import json
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Common keys used by MCP servers to hold a result list and its true total.
_LIST_KEYS = ("items", "results", "issues", "data")
_TOTAL_KEYS = ("total_count", "total", "count")

# Common parameter names a tool schema may expose to bound page size.
_LIMIT_PARAM_CANDIDATES = ("per_page", "perPage", "limit", "page_size", "pageSize")

_TRUNCATION_MARKER = "\n… [output truncated to fit the model context window]"


def wrap_tool_output(
    tool,
    *,
    max_items: int,
    max_chars: int,
    on_truncate: Optional[Callable[[str, int, int], None]] = None,
):
    """Return ``tool`` with its ``_run`` wrapped to bound argument and result size.

    Args:
        tool: A CrewAI ``BaseTool`` (both ``run()`` and ``to_structured_tool()``
            route through ``_run``, so wrapping ``_run`` covers every call path).
        max_items: Max list entries kept from a structured result.
        max_chars: Hard character budget for the returned observation string.
        on_truncate: Optional callback ``(tool_name, kept, total)`` invoked when a
            result is trimmed, so the caller can surface a user-facing note.

    Returns:
        The same tool instance, mutated in place. Returns it unchanged (with a
        warning) if the expected ``_run`` attribute is absent.
    """
    original_run = getattr(tool, "_run", None)
    if not callable(original_run):
        logger.warning("Tool %r has no callable _run; leaving unbounded", getattr(tool, "name", tool))
        return tool

    limit_param = _find_limit_param(tool, max_items)

    def bounded_run(*args: Any, **kwargs: Any) -> Any:
        # 1. Inject a fetch bound when the schema supports it and the model did
        #    not already request an equal-or-smaller page size.
        if limit_param is not None:
            existing = kwargs.get(limit_param)
            if not isinstance(existing, int) or existing > max_items:
                kwargs[limit_param] = max_items

        result = original_run(*args, **kwargs)

        # 2. Cap the observation before it becomes a prompt message.
        return _cap_result(
            result,
            tool_name=getattr(tool, "name", "tool"),
            max_items=max_items,
            max_chars=max_chars,
            on_truncate=on_truncate,
        )

    # BaseTool is a pydantic model; bypass validation to attach the bound method.
    object.__setattr__(tool, "_run", bounded_run)
    return tool


def _find_limit_param(tool, max_items: int) -> Optional[str]:
    """Return a page-size parameter name from the tool's schema, if any."""
    schema = getattr(tool, "args_schema", None)
    fields = getattr(schema, "model_fields", None) if schema is not None else None
    if not fields:
        return None
    for candidate in _LIMIT_PARAM_CANDIDATES:
        if candidate in fields:
            return candidate
    return None


def _cap_result(
    result: Any,
    *,
    tool_name: str,
    max_items: int,
    max_chars: int,
    on_truncate: Optional[Callable[[str, int, int], None]],
) -> Any:
    """Slice a structured result to max_items, then hard-cap to max_chars."""
    text = result if isinstance(result, str) else str(result)

    # Try to interpret as JSON so we can slice the list intelligently.
    parsed = None
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        parsed = None

    if isinstance(parsed, dict):
        list_key = next((k for k in _LIST_KEYS if isinstance(parsed.get(k), list)), None)
        if list_key is not None:
            items = parsed[list_key]
            total = _read_total(parsed, default=len(items))
            if len(items) > max_items:
                parsed[list_key] = items[:max_items]
                _notify(on_truncate, tool_name, max_items, total)
                text = json.dumps(parsed)
    elif isinstance(parsed, list) and len(parsed) > max_items:
        total = len(parsed)
        text = json.dumps(parsed[:max_items])
        _notify(on_truncate, tool_name, max_items, total)

    # Final hard character budget, regardless of structure.
    if len(text) > max_chars:
        _notify(on_truncate, tool_name, -1, -1)
        text = text[:max_chars] + _TRUNCATION_MARKER

    return text


def _read_total(parsed: dict, default: int) -> int:
    for key in _TOTAL_KEYS:
        value = parsed.get(key)
        if isinstance(value, int):
            return value
    return default


def _notify(cb: Optional[Callable[[str, int, int], None]], tool_name: str, kept: int, total: int) -> None:
    if cb is None:
        return
    try:
        cb(tool_name, kept, total)
    except Exception as exc:  # noqa: BLE001 - a note callback must never break tool execution
        logger.warning("Truncation callback failed: %s", exc)
