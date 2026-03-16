# Copyright 2025 IBM Corp.
# Licensed under the Apache License, Version 2.0

"""Tests for the AgentGraphCard module.

Validates:
  - EVENT_CATALOG contains all expected event types
  - Every event type has the required metadata fields
  - Categories are valid enum values
  - Terminal events have terminal=True
  - LLM events include the correct debug fields
  - Non-LLM events have empty debug_fields or only "logic"
  - build_graph_card returns a well-formed card from a compiled graph
  - The topology contains nodes, edges, and entry_node
  - Edges from the mock graph appear in the card
"""

from __future__ import annotations

import pytest
from langgraph.graph import StateGraph

from sandbox_agent.graph_card import (
    COMMON_EVENT_FIELDS,
    EVENT_CATALOG,
    TOPOLOGY_NODE_DESCRIPTIONS,
    VALID_CATEGORIES,
    build_graph_card,
)


# ---------------------------------------------------------------------------
# Expected event types (from event_schema.py NodeEventType + extensions)
# ---------------------------------------------------------------------------

EXPECTED_EVENT_TYPES = frozenset(
    {
        "planner_output",
        "executor_step",
        "thinking",
        "tool_call",
        "tool_result",
        "micro_reasoning",
        "reflector_decision",
        "reporter_output",
        "router_decision",
        "budget_update",
        "node_transition",
        "hitl_request",
    }
)

# Required keys that every catalog entry must have.
REQUIRED_ENTRY_KEYS = {"category", "description", "langgraph_nodes", "has_llm_call", "fields", "debug_fields"}

# Debug fields expected on events where has_llm_call is True.
LLM_DEBUG_FIELDS = {"system_prompt", "bound_tools", "prompt_messages", "llm_response"}


# ---------------------------------------------------------------------------
# Catalog completeness
# ---------------------------------------------------------------------------


class TestEventCatalogCompleteness:
    """EVENT_CATALOG has all expected event types."""

    def test_all_expected_types_present(self) -> None:
        assert EXPECTED_EVENT_TYPES == set(EVENT_CATALOG.keys())

    def test_no_unexpected_types(self) -> None:
        assert set(EVENT_CATALOG.keys()) - EXPECTED_EVENT_TYPES == set()


# ---------------------------------------------------------------------------
# Catalog structure
# ---------------------------------------------------------------------------


class TestEventCatalogStructure:
    """Every event type entry has the required metadata fields."""

    @pytest.mark.parametrize("event_type", sorted(EVENT_CATALOG.keys()))
    def test_required_keys(self, event_type: str) -> None:
        entry = EVENT_CATALOG[event_type]
        missing = REQUIRED_ENTRY_KEYS - set(entry.keys())
        assert not missing, f"{event_type} missing keys: {missing}"

    @pytest.mark.parametrize("event_type", sorted(EVENT_CATALOG.keys()))
    def test_category_is_valid(self, event_type: str) -> None:
        cat = EVENT_CATALOG[event_type]["category"]
        assert cat in VALID_CATEGORIES, f"{event_type} has invalid category '{cat}'"

    @pytest.mark.parametrize("event_type", sorted(EVENT_CATALOG.keys()))
    def test_langgraph_nodes_is_list(self, event_type: str) -> None:
        nodes = EVENT_CATALOG[event_type]["langgraph_nodes"]
        assert isinstance(nodes, list), f"{event_type} langgraph_nodes is not a list"

    @pytest.mark.parametrize("event_type", sorted(EVENT_CATALOG.keys()))
    def test_has_llm_call_is_bool(self, event_type: str) -> None:
        val = EVENT_CATALOG[event_type]["has_llm_call"]
        assert isinstance(val, bool), f"{event_type} has_llm_call is not bool"

    @pytest.mark.parametrize("event_type", sorted(EVENT_CATALOG.keys()))
    def test_fields_is_dict(self, event_type: str) -> None:
        fields = EVENT_CATALOG[event_type]["fields"]
        assert isinstance(fields, dict), f"{event_type} fields is not a dict"

    @pytest.mark.parametrize("event_type", sorted(EVENT_CATALOG.keys()))
    def test_debug_fields_is_dict(self, event_type: str) -> None:
        debug = EVENT_CATALOG[event_type]["debug_fields"]
        assert isinstance(debug, dict), f"{event_type} debug_fields is not a dict"


# ---------------------------------------------------------------------------
# Terminal events
# ---------------------------------------------------------------------------


class TestTerminalEvents:
    """Terminal events must have terminal=True; others must not."""

    def test_reporter_output_is_terminal(self) -> None:
        assert EVENT_CATALOG["reporter_output"].get("terminal") is True

    @pytest.mark.parametrize(
        "event_type",
        sorted(et for et in EVENT_CATALOG if et != "reporter_output"),
    )
    def test_non_terminal_events_are_not_marked(self, event_type: str) -> None:
        assert EVENT_CATALOG[event_type].get("terminal") is not True, (
            f"{event_type} should not be terminal"
        )


# ---------------------------------------------------------------------------
# Debug fields for LLM vs non-LLM events
# ---------------------------------------------------------------------------


class TestDebugFields:
    """LLM events include system_prompt/bound_tools/prompt_messages/llm_response;
    non-LLM events have empty debug_fields or only 'logic'."""

    @pytest.mark.parametrize(
        "event_type",
        sorted(et for et in EVENT_CATALOG if EVENT_CATALOG[et]["has_llm_call"]),
    )
    def test_llm_events_have_full_debug_fields(self, event_type: str) -> None:
        debug = EVENT_CATALOG[event_type]["debug_fields"]
        missing = LLM_DEBUG_FIELDS - set(debug.keys())
        assert not missing, (
            f"{event_type} (has_llm_call=True) missing debug_fields: {missing}"
        )

    @pytest.mark.parametrize(
        "event_type",
        sorted(et for et in EVENT_CATALOG if not EVENT_CATALOG[et]["has_llm_call"]),
    )
    def test_non_llm_events_debug_fields(self, event_type: str) -> None:
        debug = EVENT_CATALOG[event_type]["debug_fields"]
        if debug:
            assert set(debug.keys()) == {"logic"}, (
                f"{event_type} (has_llm_call=False) should have only 'logic' "
                f"in debug_fields, got: {set(debug.keys())}"
            )


# ---------------------------------------------------------------------------
# Common event fields
# ---------------------------------------------------------------------------


class TestCommonEventFields:
    """COMMON_EVENT_FIELDS has the required baseline fields."""

    EXPECTED_COMMON = {
        "type", "loop_id", "langgraph_node", "node_visit",
        "event_index", "model", "prompt_tokens", "completion_tokens",
    }

    def test_all_common_fields_present(self) -> None:
        assert self.EXPECTED_COMMON == set(COMMON_EVENT_FIELDS.keys())

    @pytest.mark.parametrize("field", sorted(EXPECTED_COMMON))
    def test_common_field_has_type_and_description(self, field: str) -> None:
        entry = COMMON_EVENT_FIELDS[field]
        assert "type" in entry, f"common field '{field}' missing 'type'"
        assert "description" in entry, f"common field '{field}' missing 'description'"


# ---------------------------------------------------------------------------
# Topology node descriptions
# ---------------------------------------------------------------------------


class TestTopologyNodeDescriptions:
    """TOPOLOGY_NODE_DESCRIPTIONS covers known graph nodes."""

    EXPECTED_NODES = {
        "router", "planner", "planner_tools", "step_selector",
        "executor", "tools", "reflector", "reflector_tools",
        "reflector_route", "reporter",
    }

    def test_all_graph_nodes_described(self) -> None:
        missing = self.EXPECTED_NODES - set(TOPOLOGY_NODE_DESCRIPTIONS.keys())
        assert not missing, f"Missing topology descriptions: {missing}"


# ---------------------------------------------------------------------------
# build_graph_card with a mock compiled graph
# ---------------------------------------------------------------------------


def _build_mock_graph():
    """Build a simple StateGraph that mimics the sandbox agent topology.

    Uses a minimal state (just a 'messages' key) and three nodes
    (alpha -> beta -> gamma) to exercise build_graph_card.
    """
    graph = StateGraph(dict)
    graph.add_node("alpha", lambda state: state)
    graph.add_node("beta", lambda state: state)
    graph.add_node("gamma", lambda state: state)
    graph.set_entry_point("alpha")
    graph.add_edge("alpha", "beta")
    graph.add_edge("beta", "gamma")
    graph.add_edge("gamma", "__end__")
    return graph.compile()


class TestBuildGraphCard:
    """build_graph_card returns a well-formed card."""

    @pytest.fixture()
    def card(self):
        compiled = _build_mock_graph()
        return build_graph_card(compiled, agent_id="test_agent")

    def test_card_has_id(self, card: dict) -> None:
        assert card["id"] == "test_agent"

    def test_card_has_framework(self, card: dict) -> None:
        assert card["framework"] == "langgraph"

    def test_card_has_version(self, card: dict) -> None:
        assert card["version"] == "1.0"

    def test_card_has_event_catalog(self, card: dict) -> None:
        assert "event_catalog" in card
        assert card["event_catalog"] is EVENT_CATALOG

    def test_card_has_common_event_fields(self, card: dict) -> None:
        assert "common_event_fields" in card
        assert card["common_event_fields"] is COMMON_EVENT_FIELDS

    def test_card_has_topology(self, card: dict) -> None:
        topo = card["topology"]
        assert "nodes" in topo
        assert "edges" in topo
        assert "entry_node" in topo

    def test_topology_entry_node(self, card: dict) -> None:
        assert card["topology"]["entry_node"] == "alpha"

    def test_topology_nodes_exclude_start_end(self, card: dict) -> None:
        nodes = card["topology"]["nodes"]
        assert "__start__" not in nodes
        assert "__end__" not in nodes
        assert set(nodes.keys()) == {"alpha", "beta", "gamma"}

    def test_topology_edges_include_mock_edges(self, card: dict) -> None:
        edges = card["topology"]["edges"]
        edge_pairs = {(e["source"], e["target"]) for e in edges}
        assert ("alpha", "beta") in edge_pairs
        assert ("beta", "gamma") in edge_pairs

    def test_topology_edges_exclude_start_end(self, card: dict) -> None:
        edges = card["topology"]["edges"]
        for e in edges:
            assert e["source"] not in ("__start__", "__end__")
            assert e["target"] not in ("__start__", "__end__")

    def test_topology_nodes_have_description_field(self, card: dict) -> None:
        for node_id, node_meta in card["topology"]["nodes"].items():
            assert "description" in node_meta, f"Node '{node_id}' missing description"
