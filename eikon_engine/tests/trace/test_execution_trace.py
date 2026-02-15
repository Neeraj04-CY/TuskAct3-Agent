from __future__ import annotations

from pathlib import Path

import pytest

from eikon_engine.missions.mission_executor import MissionExecutor
from eikon_engine.missions.mission_planner import MissionPlanningError
from eikon_engine.missions.mission_schema import MissionSpec, MissionSubgoal
from eikon_engine.trace.recorder import ExecutionTraceRecorder
from eikon_engine.trace.reader import read_trace


class _DummyWorker:
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


def _build_executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MissionExecutor:
    trace_dir = tmp_path / "traces"
    monkeypatch.setattr(MissionExecutor, "_build_worker", lambda self, spec: _DummyWorker())
    return MissionExecutor(
        settings={"planner": {}},
        artifacts_root=tmp_path,
        trace_recorder_factory=lambda: ExecutionTraceRecorder(storage_dir=trace_dir),
    )


def _single_trace_path(trace_dir: Path) -> Path:
    traces = list(trace_dir.glob("trace_*/trace.json"))
    assert traces, "expected a persisted trace file"
    return traces[0]


@pytest.mark.asyncio
async def test_execution_trace_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = MissionSpec(instruction="Capture data", execute=False)
    trace_dir = tmp_path / "traces"
    subgoals = [MissionSubgoal(id="sg1", description="Collect", planner_metadata={})]
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda _: subgoals)

    async def fake_run_pipeline(self, **_: object) -> dict:
        return {"completion": {"complete": True, "reason": "ok"}, "artifacts": {}}

    executor = _build_executor(tmp_path, monkeypatch)
    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", fake_run_pipeline)

    result = await executor.run_mission(spec)
    assert result.status == "complete"
    trace = read_trace(_single_trace_path(trace_dir))
    assert trace.status == "complete"
    assert trace.trace_version == "v3.1"
    assert len(trace.subgoal_traces) == 1
    assert trace.subgoal_traces[0].status == "complete"


@pytest.mark.asyncio
async def test_subgoal_trace_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = MissionSpec(instruction="Retry task", execute=False, max_retries=2)
    trace_dir = tmp_path / "traces"
    monkeypatch.setattr(
        "eikon_engine.missions.mission_planner.plan_mission",
        lambda _: [MissionSubgoal(id="sg1", description="Do it", planner_metadata={})],
    )

    attempts: dict[str, int] = {"count": 0}

    async def flaky_pipeline(self, **_: object) -> dict:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return {"completion": {"complete": False, "reason": "boom"}, "artifacts": {}, "error": "boom"}
        return {"completion": {"complete": True, "reason": "ok"}, "artifacts": {}}

    executor = _build_executor(tmp_path, monkeypatch)
    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", flaky_pipeline)

    result = await executor.run_mission(spec)
    assert result.status == "complete"
    trace = read_trace(_single_trace_path(trace_dir))
    assert len(trace.subgoal_traces) == 2
    attempts_logged = [trace_entry.attempt_number for trace_entry in trace.subgoal_traces]
    assert attempts_logged == [1, 2]
    assert trace.subgoal_traces[-1].status == "complete"


@pytest.mark.asyncio
async def test_skill_usage_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = MissionSpec(instruction="Login", execute=True)
    trace_dir = tmp_path / "traces"
    monkeypatch.setattr(
        "eikon_engine.missions.mission_planner.plan_mission",
        lambda _: [MissionSubgoal(id="sg-login", description="Login: demo", planner_metadata={})],
    )

    async def fake_run_pipeline(self, **_: object) -> dict:
        raise AssertionError("pipeline should not run when skill succeeds")

    async def fake_login_skill(self, **_: object) -> dict:
        return {"result": {"status": "success"}}

    executor = _build_executor(tmp_path, monkeypatch)
    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", fake_run_pipeline)
    monkeypatch.setattr(MissionExecutor, "_should_apply_login_skill", lambda *_, **__: True)
    monkeypatch.setattr(MissionExecutor, "_invoke_login_skill", fake_login_skill)

    result = await executor.run_mission(spec)
    assert result.status == "complete"
    trace = read_trace(_single_trace_path(trace_dir))
    assert len(trace.skills_used) == 1
    assert trace.skills_used[0].name == "login_form_skill"
    assert trace.subgoal_traces[0].skill_used == "login_form_skill"


@pytest.mark.asyncio
async def test_trace_persisted_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = MissionSpec(instruction="Plan fail", execute=False)
    trace_dir = tmp_path / "traces"

    def boom(_: MissionSpec) -> None:
        raise MissionPlanningError("planner blew up")

    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", boom)
    executor = _build_executor(tmp_path, monkeypatch)
    result = await executor.run_mission(spec)
    assert result.status == "failed"
    trace = read_trace(_single_trace_path(trace_dir))
    assert trace.status == "failed"
    assert trace.failures and trace.failures[0].failure_type == "planner_error"
