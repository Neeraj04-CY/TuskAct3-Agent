from __future__ import annotations

from eikon_engine.missions.mission_executor import MissionExecutor
from eikon_engine.missions.mission_schema import MissionSpec, MissionSubgoal


def test_planner_conflict_detected_when_score_low(monkeypatch) -> None:
    executor = MissionExecutor(settings={"learning": {"hard_floor": -0.6, "override_threshold": 0.2}})
    subgoals = [MissionSubgoal(id="sg-1", description="Login step", planner_metadata={"bucket": "login"})]
    steps = executor._to_plan_steps(subgoals)

    for step in steps:
        step["learning_score"] = -0.3
    conflicts = executor._detect_conflicts(steps)

    assert conflicts
    assert conflicts[0].learning_score < 0.0