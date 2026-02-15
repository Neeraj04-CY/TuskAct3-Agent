from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from eikon_engine.missions.mission_executor import MissionExecutor
from eikon_engine.missions.mission_schema import MissionSpec, MissionSubgoal
from eikon_engine.trace.reader import read_trace
from eikon_engine.trace.recorder import ExecutionTraceRecorder


class _StubWorker:
    def __init__(self) -> None:
        self.logger = None
        self.demo_mode = False

    async def shutdown(self) -> None:  # pragma: no cover
        return None

    def set_mission_context(self, **_: object) -> None:  # pragma: no cover
        return None

    def set_trace_context(self, **_: object) -> None:  # pragma: no cover
        return None

    def clear_trace_context(self) -> None:  # pragma: no cover
        return None

    async def execute(self, *_: object, **__: object) -> dict:  # pragma: no cover
        return {"completion": {"complete": True, "reason": "stub"}}

    async def run_skill(self, *_: object, **__: object) -> dict:  # pragma: no cover
        return {"status": "success"}


async def _noop_pipeline(*_: object, **__: object) -> dict:
    return {"completion": {"complete": True, "reason": "ok"}, "artifacts": {}}


@pytest.mark.asyncio
async def test_approval_request_written_and_resolved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec = MissionSpec(instruction="Needs approval", execute=False)
    subgoals = [
        MissionSubgoal(
            id="sg-approval",
            description="Navigate",
            planner_metadata={
                "capability_requirements": [
                    {
                        "capability_id": "missing.cap",
                        "required": True,
                        "confidence": 0.4,
                        "reason": "test",
                    }
                ]
            },
        )
    ]
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda *_: subgoals)

    trace_dir = tmp_path / "traces"
    executor = MissionExecutor(
        settings={
            "planner": {},
            "approval": {"require_approval": True, "timeout_secs": 5, "auto_approve_low_risk": False},
            "capability_enforcement": {"threshold": 0.8, "critical": 0.5},
        },
        artifacts_root=tmp_path,
        trace_recorder_factory=lambda: ExecutionTraceRecorder(storage_dir=trace_dir),
    )

    async def _approve_immediately(self, **kwargs):  # type: ignore[override]
        request_path: Path = kwargs.get("path")
        request = kwargs.get("request")
        trace_recorder = kwargs.get("trace_recorder")
        if request_path and request_path.exists():
            data = json.loads(request_path.read_text(encoding="utf-8"))
            data["state"] = "approved"
            data["reason"] = "test"
            data["resolved_at"] = datetime.now(timezone.utc).isoformat()
            request_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        if trace_recorder and request:
            trace_recorder.record_approval_resolution(
                approval_id=request.approval_id,
                subgoal_id=request.subgoal_id,
                state="approved",
                resolved_by="test",
                reason="test",
                external=False,
            )
        return "approved", "test"

    monkeypatch.setattr(MissionExecutor, "_await_approval_decision", _approve_immediately)
    monkeypatch.setattr(MissionExecutor, "_build_worker", lambda self, spec: _StubWorker())
    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", _noop_pipeline)
    monkeypatch.setattr(MissionExecutor, "_should_apply_login_skill", lambda *_, **__: False)

    result = await executor.run_mission(spec)
    assert result.status == "complete"

    approval_paths = result.summary.get("approval_requests") or []
    assert approval_paths, "approval request path missing from summary"
    request_path = Path(approval_paths[0])
    assert request_path.exists()
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert payload.get("approval_id")
    assert payload.get("state") == "approved"

    trace = read_trace(Path(result.summary["execution_trace"]))
    assert trace.approvals_requested
    assert trace.approvals_resolved
    summary_text = Path(result.summary["execution_trace_summary"]).read_text(encoding="utf-8")
    assert "Human approval requested" in summary_text


@pytest.mark.asyncio
async def test_approval_rejection_halts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec = MissionSpec(instruction="Needs approval", execute=False)
    subgoals = [
        MissionSubgoal(
            id="sg-approval",
            description="Navigate",
            planner_metadata={
                "capability_requirements": [
                    {
                        "capability_id": "missing.cap",
                        "required": True,
                        "confidence": 0.4,
                        "reason": "test",
                    }
                ]
            },
        )
    ]
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda *_: subgoals)

    trace_dir = tmp_path / "traces"
    executor = MissionExecutor(
        settings={"planner": {}, "approval": {"require_approval": True, "timeout_secs": 3}},
        artifacts_root=tmp_path,
        trace_recorder_factory=lambda: ExecutionTraceRecorder(storage_dir=trace_dir),
    )

    async def _reject(self, **kwargs):  # type: ignore[override]
        request_path: Path = kwargs.get("path")
        if request_path and request_path.exists():
            data = json.loads(request_path.read_text(encoding="utf-8"))
            data["state"] = "rejected"
            data["reason"] = "denied"
            data["resolved_at"] = datetime.now(timezone.utc).isoformat()
            request_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return "rejected", "denied"

    monkeypatch.setattr(MissionExecutor, "_await_approval_decision", _reject)
    monkeypatch.setattr(MissionExecutor, "_build_worker", lambda self, spec: _StubWorker())
    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", _noop_pipeline)
    monkeypatch.setattr(MissionExecutor, "_should_apply_login_skill", lambda *_, **__: False)

    result = await executor.run_mission(spec)
    assert result.status == "halted"
    assert result.summary.get("reason") == "approval_rejected"
    assert Path(result.summary.get("approval_request_path")).exists()
