from __future__ import annotations

from eikon_engine.planning.planner_v3 import plan_from_goal, replan_after_step


def test_replan_auth_wall_requests_restart() -> None:
    plan = plan_from_goal(
        "Log in to https://example.com/login with username demo and password pass",
        context={"credentials": {"username": "demo", "password": "pass"}},
    )
    task_id = plan["tasks"][0]["id"]

    step_result = {
        "task_id": task_id,
        "error": "Auth required",
        "meta": {"status_code": 401, "login_url": "https://example.com/login"},
        "context": {"credentials": {"username": "demo", "password": "pass"}},
    }

    partial = replan_after_step(step_result, plan)

    assert partial["meta"]["failure_type"] == "auth_wall"
    assert partial["meta"]["should_restart_task"] is True
    assert partial["meta"]["should_resume"] is False

    delta = partial["deltas"][0]
    assert delta["type"] == "replace"
    assert any(action["action"] == "fill" for action in delta["new_steps"])
