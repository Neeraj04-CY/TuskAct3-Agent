from __future__ import annotations

from eikon_engine.planning.planner_v3 import plan_from_goal, replan_after_step


def test_replan_rate_limit_contains_recovery_boundary() -> None:
    plan = plan_from_goal("Open https://example.com", context={})
    task_id = plan["tasks"][0]["id"]

    step_result = {
        "task_id": task_id,
        "error": "Rate limit hit",
        "meta": {"rate_limited": True, "retry_after_ms": 1200},
    }

    partial = replan_after_step(step_result, plan)
    delta = partial["deltas"][0]

    assert any(action.get("_recovery") for action in delta["new_steps"])
    assert any(action.get("task_boundary") for action in delta["new_steps"] if action.get("_recovery"))
