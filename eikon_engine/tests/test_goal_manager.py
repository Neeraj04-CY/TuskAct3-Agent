from __future__ import annotations

from eikon_engine.core.goal_manager import GoalManager


def test_goal_manager_parses_and_tracks_progress() -> None:
    instruction = "Log in to HerokuApp, navigate to secure area, extract title, logout"
    manager = GoalManager.parse(instruction)
    names = [goal.name for goal in manager.goals]
    assert names == [
        "open_login_page",
        "perform_login",
        "extract_secure_title",
        "logout",
    ]

    first = manager.next_goal()
    assert first is not None
    assert first.name == "open_login_page"
    assert first.status == "in_progress"

    manager.update({"goal": first.name, "completion": {"complete": True}})
    assert manager.goals[0].status == "complete"

    second = manager.next_goal()
    assert second is not None
    manager.update({"goal": second.name, "error": "bad credentials"})
    assert manager.goals[1].status == "error"
    assert manager.completion_state()["complete"] is False

    manager.update({"goal": second.name, "completion": {"complete": True}})

    for goal in manager.goals[2:]:
        manager.update({"goal": goal.name, "completion": {"complete": True}})

    assert manager.completion_state()["complete"] is True
