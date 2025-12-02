from __future__ import annotations

from eikon_engine.planning.planner_v3 import plan_from_goal, replan_after_step


def test_stability_bonus_and_penalty_fields_present() -> None:
    plan = plan_from_goal("Open https://example.com", context={})
    meta = plan["meta"]

    assert "stability_bonus" in meta
    assert meta["stability_bonus"] >= 0.0
    assert meta["replanning_penalty"] == 0.0


def test_replan_penalty_reported() -> None:
    plan = plan_from_goal("Open https://example.com/login", context={})
    task_id = plan["tasks"][0]["id"]
    step_result = {
        "task_id": task_id,
        "missing_selector": True,
        "step": {"action": "click"},
        "error": "Selector missing",
    }

    partial = replan_after_step(step_result, plan)
    assert partial["meta"]["replanning_penalty"] > 0.0
