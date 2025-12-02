from __future__ import annotations

from eikon_engine.planning.planner_v3 import plan_from_goal


def test_login_plan() -> None:
    goal = "Log in to https://example.com/login with username demo and password pass"
    context = {"credentials": {"username": "demo", "password": "pass"}}
    plan = plan_from_goal(goal, context=context)

    tasks = plan["tasks"]
    assert len(tasks) >= 2

    navigation_actions = tasks[0]["inputs"]["actions"]
    assert any(action["action"] == "navigate" and action.get("url") == "https://example.com/login" for action in navigation_actions)

    form_actions = tasks[1]["inputs"]["actions"]
    fill_actions = [action for action in form_actions if action["action"] == "fill"]
    assert fill_actions, "Expected fill action in login plan"

    fields = fill_actions[0].get("fields") or []
    assert len(fields) >= 2
    assert any(field.get("selector") == "#username" for field in fields)
    assert any(field.get("selector") == "#password" for field in fields)

    assert any(action["action"] == "click" for action in form_actions)