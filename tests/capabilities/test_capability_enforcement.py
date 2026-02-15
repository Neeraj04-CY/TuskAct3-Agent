from __future__ import annotations

import json
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

    async def shutdown(self) -> None:  # pragma: no cover - compatibility shim
        return None

    def set_mission_context(self, **_: object) -> None:  # pragma: no cover - compatibility shim
        return None

    def set_trace_context(self, **_: object) -> None:  # pragma: no cover - compatibility shim
        return None

    def clear_trace_context(self) -> None:  # pragma: no cover - compatibility shim
        return None


async def _noop_pipeline(*_: object, **__: object) -> dict:
    return {"completion": {"complete": True, "reason": "ok"}, "artifacts": {}}


@pytest.mark.asyncio
async def test_capability_enforcement_artifact_and_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = MissionSpec(instruction="Enforce capabilities", execute=False)
    subgoals = [
        MissionSubgoal(
            id="sg-warn",
            description="Navigate",
            planner_metadata={
                "capability_requirements": [
                    {
                        "capability_id": "web_navigation",
                        "required": True,
                        "confidence": 0.6,
                        "reason": "test",
                        "source": "planner",
                    }
                ]
            },
        ),
        MissionSubgoal(
            id="sg-missing",
            description="Missing capability",
            planner_metadata={
                "capability_requirements": [
                    {
                        "capability_id": "missing.capability",
                        "required": True,
                        "confidence": 0.9,
                        "reason": "test",
                    }
                ]
            },
        ),
    ]
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda *_: subgoals)

    trace_dir = tmp_path / "traces"
    executor = MissionExecutor(
        settings={"planner": {}, "capability_enforcement": {"threshold": 0.8, "critical": 0.5}},
        artifacts_root=tmp_path,
        trace_recorder_factory=lambda: ExecutionTraceRecorder(storage_dir=trace_dir),
    )

    monkeypatch.setattr(MissionExecutor, "_build_worker", lambda self, spec: _StubWorker())
    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", _noop_pipeline)
    monkeypatch.setattr(MissionExecutor, "_should_apply_login_skill", lambda *_, **__: False)

    result = await executor.run_mission(spec)
    assert result.status == "complete"

    enforcement_path = Path(result.summary["capability_enforcement_path"])
    assert enforcement_path.exists()
    payload = json.loads(enforcement_path.read_text(encoding="utf-8"))
    decisions_by_cap = {entry["capability_id"]: entry for entry in payload["decisions"]}
    assert decisions_by_cap["web_navigation"]["decision"] == "warn_only"
    assert decisions_by_cap["web_navigation"]["subgoal_id"] == "sg-warn"
    assert decisions_by_cap["missing.capability"]["decision"] == "ask_human"
    assert decisions_by_cap["missing.capability"]["missing"] is True

    trace_path = Path(result.summary["execution_trace"])
    trace = read_trace(trace_path)
    assert len(trace.capability_enforcements) == 2
    decisions = {entry.capability_id: entry for entry in trace.capability_enforcements}
    assert decisions["web_navigation"].decision == "warn_only"
    assert decisions["missing.capability"].decision == "ask_human"

    summary_text = Path(result.summary["execution_trace_summary"]).read_text(encoding="utf-8")
    assert "Capability enforcement:" in summary_text
