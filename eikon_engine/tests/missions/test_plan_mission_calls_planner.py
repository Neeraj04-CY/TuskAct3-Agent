from __future__ import annotations

from typing import Any, Dict

import pytest

from eikon_engine.missions import MissionSpec, plan_mission


@pytest.fixture
def sample_plan() -> Dict[str, Any]:
    return {
        "plan_id": "plan-1",
        "goal": "demo",
        "tasks": [
            {
                "id": "task_1",
                "tool": "BrowserWorker",
                "bucket": "navigation",
                "depends_on": [],
                "inputs": {"actions": [{"action": "navigate", "durability": "high"}]},
            }
        ],
        "meta": {"durability_summary": {"high": 1}},
    }


def test_plan_mission_uses_planner(monkeypatch: pytest.MonkeyPatch, sample_plan: Dict[str, Any]) -> None:
    captured_context: Dict[str, Any] = {}

    def fake_plan_from_goal(goal: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        captured_context.update(context or {})
        assert goal == "Login to dashboard"
        return sample_plan

    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_from_goal", fake_plan_from_goal)
    spec = MissionSpec(
        instruction="Login to dashboard",
        constraints={"password": "hunter2", "note": "safe"},
        allow_sensitive=False,
    )
    subgoals = plan_mission(spec)
    assert len(subgoals) == 1
    assert captured_context["password"] == "[REDACTED]"
    assert subgoals[0].planner_metadata["estimated_step_count"] == 1
    assert subgoals[0].planner_metadata["durability_score"] == 2.0


def test_plan_mission_injects_primary_navigation(monkeypatch: pytest.MonkeyPatch, sample_plan: Dict[str, Any]) -> None:
    monkeypatch.setattr(
        "eikon_engine.missions.mission_planner.plan_from_goal",
        lambda goal, context=None: sample_plan,
    )
    spec = MissionSpec(
        instruction="Visit https://acme.test/login and capture a screenshot",
        allow_sensitive=True,
    )
    subgoals = plan_mission(spec)
    assert len(subgoals) == len(sample_plan["tasks"])
    primary = subgoals[0]
    assert primary.planner_metadata["primary_url"] == "https://acme.test/login"
    assert primary.planner_metadata["bootstrap_actions"][0]["action"] == "navigate"
    assert primary.planner_metadata["bootstrap_actions"][0]["url"] == "https://acme.test/login"
    assert primary.description.startswith("00. navigation: navigate")
    navigation_descriptions = [sg for sg in subgoals if "navigation" in sg.description.lower()]
    assert len(navigation_descriptions) == 1
    navigation_buckets = [sg for sg in subgoals if sg.planner_metadata.get("bucket") == "navigation"]
    assert len(navigation_buckets) == 1


def test_plan_mission_injects_login_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    login_plan = {
        "plan_id": "plan-login",
        "goal": "demo",
        "tasks": [
            {
                "id": "task_form",
                "tool": "BrowserWorker",
                "bucket": "form",
                "inputs": {"actions": [{"action": "dom_presence_check", "selector": "#username"}]},
            },
            {
                "id": "task_capture",
                "tool": "BrowserWorker",
                "bucket": "capture",
                "inputs": {"actions": [{"action": "screenshot"}]},
            },
        ],
        "meta": {},
    }
    monkeypatch.setattr(
        "eikon_engine.missions.mission_planner.plan_from_goal",
        lambda goal, context=None: login_plan,
    )
    spec = MissionSpec(
        instruction="Login to https://acme.test/login and capture a screenshot",
        allow_sensitive=True,
    )
    subgoals = plan_mission(spec)
    login_subgoal = next(sg for sg in subgoals if sg.planner_metadata.get("bucket") == "login")
    actions = login_subgoal.planner_metadata["bootstrap_actions"]
    assert actions[0]["selector"] == "#username"
    assert actions[1]["selector"] == "#password"
    assert actions[-1]["action"] == "screenshot"
    assert actions[-1]["name"] == "secure_area.png"
