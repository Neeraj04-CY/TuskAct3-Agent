from __future__ import annotations

import json
from pathlib import Path

import pytest

from eikon_engine.missions.mission_executor import MissionExecutor
from eikon_engine.missions.mission_schema import MissionSpec, MissionSubgoal
from eikon_engine.trace.recorder import ExecutionTraceRecorder
from eikon_engine.trace.reader import read_trace


class _SkillWorker:
    def __init__(self) -> None:
        self.logger = None
        self.demo_mode = False
        self.skill_calls: list[tuple[str, dict]] = []
        self.learning_bias = None

    async def shutdown(self) -> None:  # pragma: no cover - compatibility shim
        return None

    def set_mission_context(self, **_: object) -> None:  # pragma: no cover - compatibility shim
        return None

    def set_trace_context(self, **_: object) -> None:  # pragma: no cover - compatibility shim
        return None

    def clear_trace_context(self) -> None:  # pragma: no cover - compatibility shim
        return None

    def set_learning_bias(self, metadata):  # pragma: no cover - compatibility shim
        self.learning_bias = metadata

    async def run_skill(self, name: str, context: dict) -> dict:
        self.skill_calls.append((name, dict(context)))
        return {"status": "success", "result": {"status": "success"}}


def _write_learning_record(root: Path, payload: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{payload['mission_id']}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


async def _run_learning_bias_demo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    logs_dir = tmp_path / "learning_logs"
    _write_learning_record(
        logs_dir,
        {
            "mission_id": "m-login-1",
            "mission_type": "login",
            "timestamp": "2026-01-13T12:00:00+00:00",
            "confidence_score": 0.91,
            "skills_used": [{"skill_name": "login_form_skill", "success": True, "steps_saved": 2}],
        },
    )
    spec = MissionSpec(instruction="Login to the portal", execute=True)
    subgoals = [MissionSubgoal(id="sg-login", description="01. Login: dom_presence_check", planner_metadata={})]
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda *_: subgoals)

    worker = _SkillWorker()
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root = tmp_path / "artifacts"

    executor = MissionExecutor(
        settings={"planner": {}, "learning": {"logs_dir": str(logs_dir)}},
        artifacts_root=artifacts_root,
        trace_recorder_factory=lambda: ExecutionTraceRecorder(storage_dir=trace_dir),
    )

    async def _fail_pipeline(*_: object, **__: object) -> dict:
        raise AssertionError("pipeline should not run when learning bias triggers login skill")

    monkeypatch.setattr(MissionExecutor, "_build_worker", lambda self, _: worker)
    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", _fail_pipeline)

    result = await executor.run_mission(spec)

    persisted = list(trace_dir.glob("trace_*/trace.json"))
    assert persisted, "expected trace artifact"
    trace = read_trace(persisted[0])
    mission_dir = Path(result.artifacts_path)
    return result, worker, trace, mission_dir


@pytest.mark.asyncio
async def test_learning_bias_triggers_login_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, worker, trace, mission_dir = await _run_learning_bias_demo(tmp_path, monkeypatch)

    assert result.status == "complete"
    assert worker.skill_calls and worker.skill_calls[0][0] == "login_form_skill"
    assert worker.learning_bias is not None

    assert trace.subgoal_traces[0].learning_bias is not None
    assert trace.skills_used[0].learning_bias["signal"]["skill"] == "login_form_skill"

    diff_path = mission_dir / "learning_diff.json"
    summary_path = mission_dir / "learning_summary.txt"
    assert diff_path.exists(), "learning diff artifact missing"
    diff_payload = json.loads(diff_path.read_text(encoding="utf-8"))
    assert diff_payload["skill_diffs"], "expected diff entry after mission learning"
    assert summary_path.exists(), "learning summary artifact missing"
    summary_text = summary_path.read_text(encoding="utf-8").strip()
    assert summary_text, "learning summary should not be empty"


@pytest.mark.asyncio
async def test_trace_contains_learning_bias_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, worker, trace, mission_dir = await _run_learning_bias_demo(tmp_path, monkeypatch)

    assert worker.learning_bias is not None
    assert trace.skills_used, "expected skill usage entries"
    usage = trace.skills_used[0]
    assert usage.learning_bias_applied is True
    assert usage.bias_snapshot is not None
    assert usage.bias_snapshot["signal"]["skill"] == "login_form_skill"
    assert (mission_dir / "learning_diff.json").exists()
