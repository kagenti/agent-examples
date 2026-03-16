# Copyright 2025 IBM Corp.
# Licensed under the Apache License, Version 2.0

"""AgentGraphCard — self-describing manifest for the agent's processing graph.

This module defines the event catalog and generates a "graph card" from
LangGraph introspection.  The graph card is a structured dict that tells
consumers (UI, backend, observability) everything they need to render the
agent's reasoning loop:

* **EVENT_CATALOG** — every event type the agent can stream, with category,
  field definitions, and debug-field metadata so the UI knows what to expect
  and how to render it.
* **COMMON_EVENT_FIELDS** — fields injected by the serializer into every
  event (type, loop_id, node_visit, event_index, etc.).
* **TOPOLOGY_NODE_DESCRIPTIONS** — human-readable descriptions for each
  LangGraph node.
* **build_graph_card()** — introspects a compiled LangGraph ``CompiledGraph``
  and returns the full card as a plain dict.
"""

from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Common fields injected into every serialized event
# ---------------------------------------------------------------------------

COMMON_EVENT_FIELDS: Dict[str, Dict[str, str]] = {
    "type": {
        "type": "str",
        "description": "Event type discriminator (one of EVENT_CATALOG keys).",
    },
    "loop_id": {
        "type": "str",
        "description": "Unique identifier for this reasoning-loop invocation.",
    },
    "langgraph_node": {
        "type": "str",
        "description": "Name of the LangGraph node that produced this event.",
    },
    "node_visit": {
        "type": "int",
        "description": "Monotonic counter incremented each time a new major node is visited.",
    },
    "event_index": {
        "type": "int",
        "description": "Global sequence number across all events in a loop (for ordering).",
    },
    "model": {
        "type": "str",
        "description": "LLM model identifier used for this event (empty if no LLM call).",
    },
    "prompt_tokens": {
        "type": "int",
        "description": "Number of prompt tokens consumed by this event's LLM call.",
    },
    "completion_tokens": {
        "type": "int",
        "description": "Number of completion tokens produced by this event's LLM call.",
    },
}

# ---------------------------------------------------------------------------
# Event catalog
# ---------------------------------------------------------------------------

#: Complete catalog of every event type the sandbox agent can stream.
#:
#: Each entry contains:
#:   category        – semantic grouping for the UI
#:   description     – what this event represents
#:   langgraph_nodes – LangGraph node names that can produce this event
#:   has_llm_call    – whether the event involves an LLM invocation
#:   terminal        – True only for the final-answer event
#:   fields          – data fields specific to this event type
#:   debug_fields    – fields available in debug / inspector mode
EVENT_CATALOG: Dict[str, Dict[str, Any]] = {
    # ── Reasoning ─────────────────────────────────────────────────────
    "planner_output": {
        "category": "reasoning",
        "description": "Planner created or revised a multi-step plan.",
        "langgraph_nodes": ["planner"],
        "has_llm_call": True,
        "fields": {
            "steps": {
                "type": "List[str]",
                "description": "Ordered list of plan step descriptions.",
            },
            "iteration": {
                "type": "int",
                "description": "Planning iteration (0 = initial, >0 = replan).",
            },
        },
        "debug_fields": {
            "system_prompt": {
                "type": "str",
                "description": "System prompt sent to the planner LLM.",
            },
            "bound_tools": {
                "type": "List[str]",
                "description": "Tool names bound to the planner LLM.",
            },
            "prompt_messages": {
                "type": "List[dict]",
                "description": "Full message history sent to the LLM.",
            },
            "llm_response": {
                "type": "str",
                "description": "Raw LLM response text.",
            },
        },
    },
    "executor_step": {
        "category": "reasoning",
        "description": "Executor selected and began working on a plan step.",
        "langgraph_nodes": ["step_selector"],
        "has_llm_call": False,
        "fields": {
            "step": {
                "type": "int",
                "description": "Current step index (1-based).",
            },
            "total_steps": {
                "type": "int",
                "description": "Total number of plan steps.",
            },
            "description": {
                "type": "str",
                "description": "Human-readable description of the current step.",
            },
            "reasoning": {
                "type": "str",
                "description": "LLM response text (up to 2000 chars).",
            },
        },
        "debug_fields": {
            "logic": {
                "type": "str",
                "description": "Step selection logic: picks current_step from plan_steps.",
            },
        },
    },
    "thinking": {
        "category": "reasoning",
        "description": (
            "Intermediate thinking iteration from a reasoning LLM "
            "(bare model, no tools)."
        ),
        "langgraph_nodes": ["planner", "executor", "reflector"],
        "has_llm_call": True,
        "fields": {
            "content": {
                "type": "str",
                "description": "Thinking text produced by the reasoning LLM.",
            },
            "iteration": {
                "type": "int",
                "description": "Thinking iteration number within this node visit.",
            },
            "total_iterations": {
                "type": "int",
                "description": "Total thinking iterations in this cycle.",
            },
        },
        "debug_fields": {
            "system_prompt": {
                "type": "str",
                "description": "System prompt for the thinking LLM.",
            },
            "bound_tools": {
                "type": "List[str]",
                "description": "Always empty — thinking LLM has no tools.",
            },
            "prompt_messages": {
                "type": "List[dict]",
                "description": "Messages sent to the thinking LLM.",
            },
            "llm_response": {
                "type": "str",
                "description": "Raw thinking response.",
            },
        },
    },
    "micro_reasoning": {
        "category": "reasoning",
        "description": (
            "Executor's intermediate LLM reasoning within a single plan step "
            "(tool-loop iteration)."
        ),
        "langgraph_nodes": ["executor"],
        "has_llm_call": True,
        "fields": {
            "content": {
                "type": "str",
                "description": "Reasoning text from the micro-reasoning LLM.",
            },
            "step": {
                "type": "int",
                "description": "Current plan step index.",
            },
            "micro_step": {
                "type": "int",
                "description": "Tool-loop iteration within the current plan step.",
            },
            "thinking_count": {
                "type": "int",
                "description": "Number of thinking iterations that preceded this reasoning.",
            },
        },
        "debug_fields": {
            "system_prompt": {
                "type": "str",
                "description": "System prompt for the micro-reasoning LLM.",
            },
            "bound_tools": {
                "type": "List[str]",
                "description": "Tool names available to the micro-reasoning LLM.",
            },
            "prompt_messages": {
                "type": "List[dict]",
                "description": "Messages sent to the micro-reasoning LLM.",
            },
            "llm_response": {
                "type": "str",
                "description": "Raw LLM response before tool extraction.",
            },
        },
    },
    # ── Execution ─────────────────────────────────────────────────────
    "tool_call": {
        "category": "execution",
        "description": "A tool was invoked by the executor or planner LLM.",
        "langgraph_nodes": ["executor", "planner"],
        "has_llm_call": False,
        "fields": {
            "step": {
                "type": "int",
                "description": "Plan step that triggered this tool call.",
            },
            "name": {
                "type": "str",
                "description": "Tool name.",
            },
            "args": {
                "type": "str",
                "description": "JSON-encoded tool arguments.",
            },
        },
        "debug_fields": {},
    },
    # ── Tool output ───────────────────────────────────────────────────
    "tool_result": {
        "category": "tool_output",
        "description": "A tool returned its result.",
        "langgraph_nodes": ["tools", "planner_tools", "reflector_tools"],
        "has_llm_call": False,
        "fields": {
            "step": {
                "type": "int",
                "description": "Plan step this result belongs to.",
            },
            "name": {
                "type": "str",
                "description": "Tool name that produced the result.",
            },
            "output": {
                "type": "str",
                "description": "Tool output (may be truncated).",
            },
        },
        "debug_fields": {},
    },
    # ── Decision ──────────────────────────────────────────────────────
    "reflector_decision": {
        "category": "decision",
        "description": (
            "Reflector reviewed execution and decided: continue, replan, or done."
        ),
        "langgraph_nodes": ["reflector"],
        "has_llm_call": True,
        "fields": {
            "decision": {
                "type": "str",
                "description": "Routing decision.",
                "enum": ["continue", "replan", "done"],
            },
            "assessment": {
                "type": "str",
                "description": "Full reflection assessment text.",
            },
            "iteration": {
                "type": "int",
                "description": "Reflect-execute loop iteration.",
            },
        },
        "debug_fields": {
            "system_prompt": {
                "type": "str",
                "description": "System prompt for the reflector LLM.",
            },
            "bound_tools": {
                "type": "List[str]",
                "description": "Read-only tools bound to the reflector.",
            },
            "prompt_messages": {
                "type": "List[dict]",
                "description": "Messages sent to the reflector LLM.",
            },
            "llm_response": {
                "type": "str",
                "description": "Raw reflector LLM output.",
            },
        },
    },
    "router_decision": {
        "category": "decision",
        "description": "Router decided whether to plan from scratch or resume execution.",
        "langgraph_nodes": ["router"],
        "has_llm_call": False,
        "fields": {
            "route": {
                "type": "str",
                "description": "Chosen route.",
                "enum": ["plan", "resume"],
            },
            "plan_status": {
                "type": "str",
                "description": "Current plan status at time of routing.",
            },
        },
        "debug_fields": {
            "logic": {
                "type": "str",
                "description": (
                    "Routing logic: checks plan_status to decide resume vs plan."
                ),
            },
        },
    },
    # ── Terminal ──────────────────────────────────────────────────────
    "reporter_output": {
        "category": "terminal",
        "description": "Reporter generated the final answer for the user.",
        "langgraph_nodes": ["reporter"],
        "has_llm_call": True,
        "terminal": True,
        "fields": {
            "content": {
                "type": "str",
                "description": "Final answer content (markdown).",
            },
        },
        "debug_fields": {
            "system_prompt": {
                "type": "str",
                "description": "System prompt for the reporter LLM.",
            },
            "bound_tools": {
                "type": "List[str]",
                "description": "Tools available to the reporter (for citations).",
            },
            "prompt_messages": {
                "type": "List[dict]",
                "description": "Messages sent to the reporter LLM.",
            },
            "llm_response": {
                "type": "str",
                "description": "Raw reporter LLM output.",
            },
        },
    },
    # ── Meta ──────────────────────────────────────────────────────────
    "budget_update": {
        "category": "meta",
        "description": "Budget tracking update (tokens consumed, wall-clock time).",
        "langgraph_nodes": [],
        "has_llm_call": False,
        "fields": {
            "tokens_used": {
                "type": "int",
                "description": "Total tokens consumed so far.",
            },
            "tokens_budget": {
                "type": "int",
                "description": "Maximum token budget.",
            },
            "wall_clock_s": {
                "type": "float",
                "description": "Elapsed wall-clock seconds.",
            },
            "max_wall_clock_s": {
                "type": "float",
                "description": "Maximum allowed wall-clock seconds.",
            },
        },
        "debug_fields": {},
    },
    "node_transition": {
        "category": "meta",
        "description": (
            "Internal marker indicating a graph-level transition between nodes."
        ),
        "langgraph_nodes": [],
        "has_llm_call": False,
        "fields": {
            "from_node": {
                "type": "str",
                "description": "Node the transition originates from.",
            },
            "to_node": {
                "type": "str",
                "description": "Node the transition goes to.",
            },
        },
        "debug_fields": {},
    },
    # ── Interaction ───────────────────────────────────────────────────
    "hitl_request": {
        "category": "interaction",
        "description": (
            "Human-in-the-loop approval request — the executor is pausing "
            "to ask the user before proceeding."
        ),
        "langgraph_nodes": ["executor"],
        "has_llm_call": False,
        "fields": {
            "tool_name": {
                "type": "str",
                "description": "Tool that requires approval.",
            },
            "args": {
                "type": "str",
                "description": "JSON-encoded tool arguments pending approval.",
            },
            "reason": {
                "type": "str",
                "description": "Why the agent is requesting approval.",
            },
        },
        "debug_fields": {},
    },
}

# Valid category values (mirrors the set used in EVENT_CATALOG).
VALID_CATEGORIES = frozenset(
    {
        "reasoning",
        "execution",
        "tool_output",
        "decision",
        "terminal",
        "meta",
        "interaction",
    }
)

# ---------------------------------------------------------------------------
# LangGraph topology node descriptions
# ---------------------------------------------------------------------------

#: Human-readable description for each node in the compiled graph.
TOPOLOGY_NODE_DESCRIPTIONS: Dict[str, str] = {
    "router": (
        "Entry node — decides whether to create a new plan or resume execution "
        "of an existing plan."
    ),
    "planner": (
        "Creates or revises a multi-step plan using an LLM with planning tools "
        "(glob, grep, file_read, file_write)."
    ),
    "planner_tools": (
        "Executes tool calls issued by the planner (workspace inspection, "
        "plan persistence)."
    ),
    "step_selector": (
        "Picks the next plan step to execute and prepares the executor context."
    ),
    "executor": (
        "Executes the current plan step using an LLM with the full tool suite "
        "(shell, files, grep, glob, web_fetch, explore, delegate)."
    ),
    "tools": (
        "Executes tool calls issued by the executor."
    ),
    "reflector": (
        "Reviews execution results and decides whether to continue, replan, "
        "or declare done. Uses read-only tools (glob, grep, file_read)."
    ),
    "reflector_tools": (
        "Executes read-only tool calls issued by the reflector for verification."
    ),
    "reflector_route": (
        "Pass-through node that routes the reflector's decision to the next node "
        "(reporter, step_selector, or planner)."
    ),
    "reporter": (
        "Generates the final user-facing answer by synthesizing all execution "
        "results. May invoke tools internally for citation verification."
    ),
}


# ---------------------------------------------------------------------------
# Graph card builder
# ---------------------------------------------------------------------------


def build_graph_card(
    compiled: Any,
    agent_id: str = "sandbox_agent",
) -> Dict[str, Any]:
    """Build the AgentGraphCard from a compiled LangGraph.

    Parameters
    ----------
    compiled:
        A ``CompiledStateGraph`` (or any object whose ``.get_graph()`` returns
        a ``Graph`` with ``.nodes`` and ``.edges``).
    agent_id:
        Identifier for the agent (used in the card's ``id`` field).

    Returns
    -------
    dict
        A plain dict with keys:
        - ``id`` — agent identifier
        - ``framework`` — always ``"langgraph"``
        - ``version`` — card schema version
        - ``event_catalog`` — the full ``EVENT_CATALOG``
        - ``common_event_fields`` — the ``COMMON_EVENT_FIELDS`` dict
        - ``topology`` — ``{nodes, edges, entry_node}``
    """
    graph = compiled.get_graph()

    # ── Nodes ─────────────────────────────────────────────────────────
    raw_nodes: List[str] = [
        node_id
        for node_id in graph.nodes
        if node_id not in ("__start__", "__end__")
    ]
    nodes: Dict[str, Dict[str, str]] = {}
    for node_id in raw_nodes:
        nodes[node_id] = {
            "description": TOPOLOGY_NODE_DESCRIPTIONS.get(node_id, ""),
        }

    # ── Edges ─────────────────────────────────────────────────────────
    edges: List[Dict[str, str]] = []
    for edge in graph.edges:
        source = edge.source if hasattr(edge, "source") else edge[0]
        target = edge.target if hasattr(edge, "target") else edge[1]
        # Skip __start__ / __end__ for cleaner topology
        if source in ("__start__", "__end__") or target in ("__start__", "__end__"):
            continue
        edges.append({"source": source, "target": target})

    # ── Entry node ────────────────────────────────────────────────────
    # The entry node is the first node reachable from __start__.
    entry_node: str = ""
    for edge in graph.edges:
        src = edge.source if hasattr(edge, "source") else edge[0]
        tgt = edge.target if hasattr(edge, "target") else edge[1]
        if src == "__start__":
            entry_node = tgt
            break

    return {
        "id": agent_id,
        "framework": "langgraph",
        "version": "1.0",
        "event_catalog": EVENT_CATALOG,
        "common_event_fields": COMMON_EVENT_FIELDS,
        "topology": {
            "nodes": nodes,
            "edges": edges,
            "entry_node": entry_node,
        },
    }
