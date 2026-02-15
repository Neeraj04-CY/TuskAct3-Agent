from __future__ import annotations

import json
from pathlib import Path

import pytest

from eikon_engine.learning.recorder import LearningRecorder
from eikon_engine.missions.mission_executor import MissionExecutor
from eikon_engine.missions.mission_schema import MissionSpec, MissionSubgoal
from eikon_engine.trace.recorder import ExecutionTraceRecorder


class _FailingWorker:
    def __init__(self) -> None:
        self.logger = None
        self.shutdown_called = False
        self.demo_mode = False

    async def shutdown(self) -> None:
        self.shutdown_called = True

    def set_mission_context(self, **_: object) -> None:
        return None

    def set_trace_context(self, **_: object) -> None:
        return None

    def clear_trace_context(self) -> None:
        return None

    async def run_skill(self, *_: object, **__: object) -> dict:
        raise RuntimeError("Browser session is not available")


@pytest.mark.asyncio
async def test_learning_records_failure_when_browser_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = MissionSpec(instruction="Login mission", execute=True)
    trace_dir = tmp_path / "traces"
    mission_dir = tmp_path / "artifacts"

    monkeypatch.setattr(
        "eikon_engine.missions.mission_planner.plan_mission",
        lambda _: [MissionSubgoal(id="sg1", description="Login: demo", planner_metadata={})],
    )

    async def pipeline_should_not_run(self, **_: object) -> dict:
        raise AssertionError("pipeline should not run when browser is unavailable")

    executor = MissionExecutor(
        settings={"planner": {}},
        artifacts_root=mission_dir,
        trace_recorder_factory=lambda: ExecutionTraceRecorder(storage_dir=trace_dir),
    )
    executor.learning_recorder = LearningRecorder(output_dir=tmp_path / "learning_logs")

    monkeypatch.setattr(MissionExecutor, "_build_worker", lambda self, spec: _FailingWorker())
    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", pipeline_should_not_run)
    monkeypatch.setattr(MissionExecutor, "_should_apply_login_skill", lambda *_, **__: True)

    result = await executor.run_mission(spec)

    assert result.status == "failed"
    assert any("browser session" in (sub.error or "").lower() for sub in result.subgoal_results)

    learning_path = tmp_path / "learning_logs" / f"{result.mission_id}.json"
    assert learning_path.exists()
    payload = json.loads(learning_path.read_text())
    assert payload.get("outcome") == "failure"
    assert payload.get("confidence_score", 1.0) < 0.5
    assert payload.get("skills_used") is not None
