from __future__ import annotations

from eikon_engine.planning.planner_v3 import plan_from_goal


def test_multigoal_plan_merging() -> None:
    goal = "Open https://example.com/login then log in"
    context = {"credentials": {"username": "demo", "password": "pass"}}

    plan = plan_from_goal(goal, context=context)

    # Expect meta information to capture subplans
    subplans = plan["meta"].get("subplans")
    assert subplans and len(subplans) == 2

    tasks = plan["tasks"]
    assert len(tasks) >= 3

    # Ensure tasks remain ordered and navigation stays first
    assert tasks[0]["inputs"]["actions"][0]["action"] == "navigate"
    for idx in range(1, len(tasks)):
        assert tasks[idx]["depends_on"] == [tasks[idx - 1]["id"]]

    # Ensure the final non-recovery task belongs to login flow (click or fill)
    for task in reversed(tasks):
        if not any(action.get("_recovery") for action in task["inputs"]["actions"]):
            assert any(action["action"] in {"fill", "click"} for action in task["inputs"]["actions"])
            break
    else:
        raise AssertionError("Expected at least one non-recovery task with form actions")
