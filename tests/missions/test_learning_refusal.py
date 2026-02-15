from __future__ import annotations

import pytest

from eikon_engine.missions.mission_executor import MissionExecutor
from eikon_engine.missions.mission_schema import MissionSpec, MissionSubgoal


@pytest.mark.asyncio
async def test_mission_refused_by_learning(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = MissionSpec(instruction="Login demo", execute=True)
    subgoals = [MissionSubgoal(id="sg-1", description="Login", planner_metadata={"bucket": "login"})]
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda *_: subgoals)

    executor = MissionExecutor(settings={"learning": {"hard_floor": -0.6, "override_threshold": 0.1}})

    # Force learning score below hard floor
    monkeypatch.setattr(executor, "_compute_learning_score", lambda *_: -0.9)

    result = await executor.run_mission(spec)

    assert result.status == "refused_by_learning"
    assert result.summary.get("reason") == "learning_refusal"
