from __future__ import annotations

from eikon_engine.planning.planner_v3 import plan_from_goal, replan_after_step


def test_replan_navigation_loop_inserts_navigation() -> None:
    plan = plan_from_goal("Open https://example.com/login", context={})
    task_id = plan["tasks"][0]["id"]

    step_result = {
        "task_id": task_id,
        "error": "Too many redirects",
        "meta": {"loop_url": "https://example.com/login", "wait_selector": "#main"},
    }

    partial = replan_after_step(step_result, plan)
    assert partial["meta"]["failure_type"] == "navigation_loop"
    assert partial["meta"]["should_resume"] is True

    delta = partial["deltas"][0]
    assert delta["type"] == "insert"
    assert any(action["action"] == "navigate" for action in delta["new_steps"])
