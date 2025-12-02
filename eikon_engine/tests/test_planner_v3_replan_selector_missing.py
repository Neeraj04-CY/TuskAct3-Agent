from __future__ import annotations

from eikon_engine.planning.planner_v3 import plan_from_goal, replan_after_step


def test_replan_selector_missing_replaces_task() -> None:
    plan = plan_from_goal(
        "Log in to https://example.com/login with username demo and password pass",
        context={"credentials": {"username": "demo", "password": "pass"}},
    )

    failing_task = plan["tasks"][1]["id"] if len(plan["tasks"]) > 1 else plan["tasks"][0]["id"]
    step_result = {
        "task_id": failing_task,
        "step": {"action": "click", "selector": "#login"},
        "error": "Selector #login not found",
        "missing_selector": True,
    }

    partial = replan_after_step(step_result, plan)

    assert partial["meta"]["failure_type"] == "selector_missing"
    assert partial["meta"]["should_resume"] is True
    assert partial["meta"]["should_restart_task"] is False
    assert partial["deltas"], "Expected delta instructions"

    delta = partial["deltas"][0]
    assert delta["type"] == "replace"
    assert delta["target_task"] == failing_task
    assert any(action.get("_precheck") for action in delta["new_steps"])
