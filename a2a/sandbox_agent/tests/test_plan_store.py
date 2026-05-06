"""Tests for the append-only PlanStore."""

import pytest
from sandbox_agent.plan_store import (
    add_alternative_subplan,
    add_steps,
    all_terminal,
    create_plan,
    done_count,
    get_active_substep,
    get_current_step,
    set_step_status,
    set_substep_status,
    step_count,
    to_flat_plan,
    to_flat_plan_steps,
)


class TestCreatePlan:
    def test_creates_indexed_steps(self) -> None:
        plan = create_plan(["Clone repo", "Analyze logs", "Write report"])
        assert step_count(plan) == 3
        assert plan["steps"]["1"]["description"] == "Clone repo"
        assert plan["steps"]["2"]["description"] == "Analyze logs"
        assert plan["steps"]["3"]["description"] == "Write report"

    def test_first_step_is_running(self) -> None:
        plan = create_plan(["Step 1", "Step 2"])
        assert plan["steps"]["1"]["status"] == "running"
        assert plan["steps"]["2"]["status"] == "pending"

    def test_each_step_has_subplan_a(self) -> None:
        plan = create_plan(["Step 1"])
        step = plan["steps"]["1"]
        assert "a" in step["subplans"]
        assert step["active_subplan"] == "a"
        assert step["subplans"]["a"]["created_by"] == "planner"

    def test_subplan_has_substep(self) -> None:
        plan = create_plan(["Clone repo"])
        substeps = plan["steps"]["1"]["subplans"]["a"]["substeps"]
        assert "1" in substeps
        assert substeps["1"]["description"] == "Clone repo"
        assert substeps["1"]["status"] == "pending"

    def test_empty_plan(self) -> None:
        plan = create_plan([])
        assert step_count(plan) == 0
        assert plan["version"] == 1


class TestAddSteps:
    def test_add_after_all_terminal(self) -> None:
        plan = create_plan(["Step 1"])
        plan = set_step_status(plan, "1", "done")
        plan = add_steps(plan, ["Step 2", "Step 3"], creator="replanner")
        assert step_count(plan) == 3
        assert plan["steps"]["2"]["description"] == "Step 2"
        assert plan["steps"]["2"]["status"] == "running"
        assert plan["steps"]["3"]["status"] == "pending"

    def test_rejects_non_replanner(self) -> None:
        plan = create_plan(["Step 1"])
        plan = set_step_status(plan, "1", "done")
        with pytest.raises(ValueError, match="Only replanner"):
            add_steps(plan, ["Step 2"], creator="planner")

    def test_rejects_when_active_steps_exist(self) -> None:
        plan = create_plan(["Step 1", "Step 2"])
        with pytest.raises(ValueError, match="still active"):
            add_steps(plan, ["Step 3"], creator="replanner")

    def test_allows_after_mixed_terminal(self) -> None:
        plan = create_plan(["Step 1", "Step 2"])
        plan = set_step_status(plan, "1", "done")
        plan = set_step_status(plan, "2", "failed")
        plan = add_steps(plan, ["Step 3"], creator="replanner")
        assert step_count(plan) == 3

    def test_does_not_mutate_original(self) -> None:
        plan = create_plan(["Step 1"])
        plan = set_step_status(plan, "1", "done")
        new_plan = add_steps(plan, ["Step 2"], creator="replanner")
        assert step_count(plan) == 1  # original unchanged
        assert step_count(new_plan) == 2


class TestAddAlternativeSubplan:
    def test_creates_subplan_b(self) -> None:
        plan = create_plan(["Step 1"])
        plan = set_step_status(plan, "1", "failed")
        plan["steps"]["1"]["subplans"]["a"]["status"] = "failed"
        new_plan, key = add_alternative_subplan(plan, "1", ["Alt approach 1", "Alt approach 2"])
        assert key == "b"
        step = new_plan["steps"]["1"]
        assert "b" in step["subplans"]
        assert step["active_subplan"] == "b"
        assert step["subplans"]["b"]["created_by"] == "replanner"
        assert len(step["subplans"]["b"]["substeps"]) == 2

    def test_creates_subplan_c(self) -> None:
        plan = create_plan(["Step 1"])
        plan, _ = add_alternative_subplan(plan, "1", ["Alt B"])
        plan, key = add_alternative_subplan(plan, "1", ["Alt C"])
        assert key == "c"

    def test_switches_active_subplan(self) -> None:
        plan = create_plan(["Step 1"])
        plan, key = add_alternative_subplan(plan, "1", ["Alt"])
        assert plan["steps"]["1"]["active_subplan"] == key

    def test_rejects_missing_step(self) -> None:
        plan = create_plan(["Step 1"])
        with pytest.raises(ValueError, match="not found"):
            add_alternative_subplan(plan, "99", ["Alt"])

    def test_does_not_mutate_original(self) -> None:
        plan = create_plan(["Step 1"])
        new_plan, _ = add_alternative_subplan(plan, "1", ["Alt"])
        assert len(plan["steps"]["1"]["subplans"]) == 1
        assert len(new_plan["steps"]["1"]["subplans"]) == 2


class TestStatusUpdates:
    def test_set_step_running(self) -> None:
        plan = create_plan(["Step 1", "Step 2"])
        plan = set_step_status(plan, "2", "running")
        assert plan["steps"]["2"]["status"] == "running"

    def test_set_step_done(self) -> None:
        plan = create_plan(["Step 1"])
        plan = set_step_status(plan, "1", "done")
        assert plan["steps"]["1"]["status"] == "done"

    def test_terminal_status_is_sticky(self) -> None:
        plan = create_plan(["Step 1"])
        plan = set_step_status(plan, "1", "done")
        plan = set_step_status(plan, "1", "running")  # should be ignored
        assert plan["steps"]["1"]["status"] == "done"

    def test_updates_active_subplan_status(self) -> None:
        plan = create_plan(["Step 1"])
        plan = set_step_status(plan, "1", "done")
        assert plan["steps"]["1"]["subplans"]["a"]["status"] == "done"

    def test_set_substep_status(self) -> None:
        plan = create_plan(["Step 1"])
        plan = set_substep_status(plan, "1", "1", "done", result_summary="cloned OK")
        ss = plan["steps"]["1"]["subplans"]["a"]["substeps"]["1"]
        assert ss["status"] == "done"
        assert ss["result_summary"] == "cloned OK"

    def test_set_substep_tool_calls(self) -> None:
        plan = create_plan(["Step 1"])
        plan = set_substep_status(plan, "1", "1", "running", tool_calls=["shell", "grep"])
        ss = plan["steps"]["1"]["subplans"]["a"]["substeps"]["1"]
        assert ss["tool_calls"] == ["shell", "grep"]


class TestQueries:
    def test_get_current_step(self) -> None:
        plan = create_plan(["Step 1", "Step 2", "Step 3"])
        key, step = get_current_step(plan)
        assert key == "1"
        assert step["description"] == "Step 1"

    def test_get_current_step_after_done(self) -> None:
        plan = create_plan(["Step 1", "Step 2"])
        plan = set_step_status(plan, "1", "done")
        plan = set_step_status(plan, "2", "running")
        key, step = get_current_step(plan)
        assert key == "2"

    def test_get_current_step_all_done(self) -> None:
        plan = create_plan(["Step 1"])
        plan = set_step_status(plan, "1", "done")
        assert get_current_step(plan) is None

    def test_get_active_substep(self) -> None:
        plan = create_plan(["Step 1"])
        result = get_active_substep(plan, "1")
        assert result is not None
        key, ss = result
        assert key == "1"

    def test_done_count(self) -> None:
        plan = create_plan(["S1", "S2", "S3"])
        plan = set_step_status(plan, "1", "done")
        plan = set_step_status(plan, "2", "done")
        assert done_count(plan) == 2

    def test_all_terminal(self) -> None:
        plan = create_plan(["S1", "S2"])
        assert not all_terminal(plan)
        plan = set_step_status(plan, "1", "done")
        plan = set_step_status(plan, "2", "failed")
        assert all_terminal(plan)


class TestFlatConversion:
    def test_to_flat_plan(self) -> None:
        plan = create_plan(["Clone repo", "Analyze", "Report"])
        flat = to_flat_plan(plan)
        assert flat == ["Clone repo", "Analyze", "Report"]

    def test_to_flat_plan_steps(self) -> None:
        plan = create_plan(["Clone repo", "Analyze"])
        plan = set_step_status(plan, "1", "done")
        flat = to_flat_plan_steps(plan)
        assert len(flat) == 2
        assert flat[0]["index"] == 0
        assert flat[0]["status"] == "done"
        assert flat[0]["active_subplan"] == "a"
        assert flat[0]["alternative_count"] == 0

    def test_flat_plan_steps_with_alternatives(self) -> None:
        plan = create_plan(["Step 1"])
        plan, _ = add_alternative_subplan(plan, "1", ["Alt approach"])
        flat = to_flat_plan_steps(plan)
        assert flat[0]["alternative_count"] == 1
        assert flat[0]["active_subplan"] == "b"
