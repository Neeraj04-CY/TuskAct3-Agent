from __future__ import annotations

import json
from pathlib import Path

import pytest

from eikon_engine.capabilities.registry import (
    CAPABILITY_REGISTRY,
    all_capabilities,
    capabilities_for_skill,
    get_capability,
)
from eikon_engine.missions.mission_executor import MissionExecutor
from eikon_engine.missions.mission_schema import MissionSpec, MissionSubgoal
from eikon_engine.trace.reader import read_trace
from eikon_engine.trace.recorder import ExecutionTraceRecorder


def test_registry_lookup_and_skill_mapping() -> None:
    auth_cap = get_capability("auth.login")
    listing_cap = get_capability("data.listing_extraction")

    assert auth_cap is not None and auth_cap.id == "auth.login"
    assert listing_cap is not None and listing_cap.id == "data.listing_extraction"

    caps = all_capabilities()
    assert len(caps) >= 2
    skills = capabilities_for_skill("login_form_skill")
    assert any(cap.id == "auth.login" for cap in skills)


class _StubWorker:
    def __init__(self) -> None:
        self.logger = None
        self.demo_mode = False

    async def shutdown(self) -> None:  # pragma: no cover - compatibility shim
        return None

    def set_mission_context(self, **_: object) -> None:  # pragma: no cover - compatibility shim
        return None

    def set_trace_context(self, **_: object) -> None:  # pragma: no cover - compatibility shim
        return None

    def clear_trace_context(self) -> None:  # pragma: no cover - compatibility shim
        return None


async def _login_success(*_: object, **__: object) -> dict:
    return {"result": {"status": "success"}}


async def _noop_pipeline(*_: object, **__: object) -> dict:
    return {"completion": {"complete": True, "reason": "ok"}, "artifacts": {}}


def _single_trace_path(trace_dir: Path) -> Path:
    traces = list(trace_dir.glob("trace_*/trace.json"))
    assert traces, "expected a persisted trace"
    return traces[0]


@pytest.mark.asyncio
async def test_capability_usage_recorded_on_successful_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = MissionSpec(instruction="Login mission", execute=True)
    subgoals = [MissionSubgoal(id="sg-login", description="Login", planner_metadata={})]
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda *_: subgoals)

    trace_dir = tmp_path / "traces"
    artifacts_root = tmp_path / "artifacts"
    executor = MissionExecutor(
        settings={"planner": {}},
        artifacts_root=artifacts_root,
        trace_recorder_factory=lambda: ExecutionTraceRecorder(storage_dir=trace_dir),
    )

    monkeypatch.setattr(MissionExecutor, "_build_worker", lambda self, spec: _StubWorker())
    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", _noop_pipeline)
    monkeypatch.setattr(MissionExecutor, "_should_apply_login_skill", lambda *_, **__: True)
    monkeypatch.setattr(MissionExecutor, "_invoke_login_skill", _login_success)

    result = await executor.run_mission(spec)
    assert result.status == "complete"

    trace = read_trace(_single_trace_path(trace_dir))
    assert trace.capabilities_used
    usage = trace.capabilities_used[0]
    assert usage.capability_id == "auth.login"
    assert usage.skill_id == "login_form_skill"
    assert trace.subgoal_traces[0].capabilities_used

    summary_path = Path(result.summary["execution_trace_summary"])
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Capabilities used" in summary_text


@pytest.mark.asyncio
async def test_unknown_skill_records_no_capability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = MissionSpec(instruction="Unknown skill mission", execute=True)
    subgoals = [MissionSubgoal(id="sg-unknown", description="Unknown", planner_metadata={})]
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda *_: subgoals)

    trace_dir = tmp_path / "traces"
    artifacts_root = tmp_path / "artifacts"
    executor = MissionExecutor(
        settings={"planner": {}},
        artifacts_root=artifacts_root,
        trace_recorder_factory=lambda: ExecutionTraceRecorder(storage_dir=trace_dir),
    )

    monkeypatch.setattr(MissionExecutor, "_build_worker", lambda self, spec: _StubWorker())
    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", _noop_pipeline)
    monkeypatch.setattr(MissionExecutor, "_should_apply_login_skill", lambda *_, **__: True)
    monkeypatch.setattr(MissionExecutor, "_invoke_login_skill", _login_success)
    monkeypatch.setattr("eikon_engine.capabilities.registry.capabilities_for_skill", lambda skill: [])

    result = await executor.run_mission(spec)
    assert result.status == "complete"

    trace = read_trace(_single_trace_path(trace_dir))
    assert trace.capabilities_used == []
    assert trace.subgoal_traces[0].capabilities_used == []

    summary_path = Path(result.summary["execution_trace_summary"])
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Capabilities used: none" in summary_text
