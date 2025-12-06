from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from eikon_engine.missions.mission_executor import MissionExecutor
from eikon_engine.missions.mission_schema import MissionSpec, MissionSubgoal


@pytest.mark.asyncio
async def test_mission_executor_retry_on_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = MissionSpec(instruction="Retry until success", execute=True, max_retries=2)
    monkeypatch.setattr(
        "eikon_engine.missions.mission_planner.plan_mission",
        lambda _: [MissionSubgoal(id="sg1", description="Try once", planner_metadata={})],
    )

    calls: Dict[str, int] = {"count": 0}

    async def fake_run_pipeline(self, **_: object) -> dict:  # noqa: ARG001 - self is required for monkeypatching
        calls["count"] += 1
        if calls["count"] == 1:
            return {"completion": {"complete": False, "reason": "boom"}, "artifacts": {}}
        return {"completion": {"complete": True, "reason": "ok"}, "artifacts": {}}

    sleep_calls: list[float] = []

    async def fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)

    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", fake_run_pipeline)
    executor = MissionExecutor(
        settings={"planner": {}, "logging": {"artifact_root": str(tmp_path)}},
        artifacts_root=tmp_path,
        sleep_fn=fake_sleep,
    )
    result = await executor.run_mission(spec)
    assert result.status == "complete"
    assert sleep_calls == [1.0]
    assert result.subgoal_results[0].attempts == 2
