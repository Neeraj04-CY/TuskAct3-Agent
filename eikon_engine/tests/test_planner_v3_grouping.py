from __future__ import annotations

from eikon_engine.planning.planner_v3 import plan_from_goal


def test_task_bucket_labels() -> None:
    goal = "Log in to https://example.com/login with username demo and password pass"
    context = {"credentials": {"username": "demo", "password": "pass"}}

    plan = plan_from_goal(goal, context=context)
    tasks = plan["tasks"]

    assert tasks[0]["bucket"] == "navigation"
    assert any(task["bucket"] == "form" for task in tasks)

    for task in tasks:
        assert task["bucket"] in {"navigation", "form", "extraction", "wait", "screenshot", "misc"}
