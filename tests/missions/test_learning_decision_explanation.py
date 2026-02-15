from __future__ import annotations

import json
from pathlib import Path

import pytest

from eikon_engine.missions.mission_executor import MissionExecutor
from eikon_engine.missions.mission_schema import MissionSpec, MissionSubgoal
from eikon_engine.trace.recorder import ExecutionTraceRecorder


class _StubWorker:
    def __init__(self) -> None:
        self.logger = None
        self.demo_mode = False
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

    async def run_skill(self, *_: object, **__: object) -> dict:
        return {"status": "success", "result": {"status": "success"}}


async def _success_pipeline(*_: object, **__: object) -> dict:
    return {"completion": {"complete": True, "reason": "ok"}, "artifacts": {}}


def _write_learning_record(root: Path, payload: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{payload['mission_id']}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mission_result_paths(mission_dir: Path) -> tuple[Path, Path]:
    mission_result = json.loads((mission_dir / "mission_result.json").read_text(encoding="utf-8"))
    summary_path = Path(mission_result["summary"]["execution_trace_summary"])
    explanation_path = mission_dir / "learning_decision_explanation.json"
    return summary_path, explanation_path


@pytest.mark.asyncio
async def test_learning_override_writes_explanation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = MissionSpec(instruction="Demo override", execute=True)
    subgoals = [
        MissionSubgoal(id="sg-risky", description="01. risky", planner_metadata={"bucket": "login"}),
        MissionSubgoal(id="sg-safe", description="02. safe", planner_metadata={"bucket": "login"}),
    ]
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda *_: subgoals)

    worker = _StubWorker()
    trace_dir = tmp_path / "traces"
    executor = MissionExecutor(
        settings={"learning": {"override_threshold": 0.0, "hard_floor": -0.6}},
        artifacts_root=tmp_path / "artifacts",
        trace_recorder_factory=lambda: ExecutionTraceRecorder(storage_dir=trace_dir),
    )

    monkeypatch.setattr(MissionExecutor, "_build_worker", lambda self, spec: worker)
    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", _success_pipeline)
    monkeypatch.setattr(executor, "_compute_learning_score", lambda step: -0.1 if "risky" in step.get("description", "") else 0.7)

    result = await executor.run_mission(spec)

    mission_dir = Path(result.artifacts_path)
    summary_path, explanation_path = _mission_result_paths(mission_dir)
    payload = json.loads(explanation_path.read_text(encoding="utf-8"))

    assert result.status == "complete"
    assert explanation_path.exists()
    assert payload["mission_id"] == spec.id
    assert payload["decision_type"] == "override"
    assert payload["final_resolution"] == "override_applied"
    assert payload["triggering_signals"]

    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Learning decision explanation:" in summary_text


@pytest.mark.asyncio
async def test_learning_refusal_writes_explanation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = MissionSpec(instruction="Refusal mission", execute=True)
    subgoals = [MissionSubgoal(id="sg-1", description="Login", planner_metadata={"bucket": "login"})]
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda *_: subgoals)

    executor = MissionExecutor(
        settings={"learning": {"override_threshold": 0.1, "hard_floor": -0.6}},
        artifacts_root=tmp_path / "artifacts",
        trace_recorder_factory=lambda: ExecutionTraceRecorder(storage_dir=tmp_path / "traces"),
    )
    monkeypatch.setattr(executor, "_compute_learning_score", lambda *_, **__: -0.9)

    result = await executor.run_mission(spec)

    mission_dir = Path(result.artifacts_path)
    summary_path, explanation_path = _mission_result_paths(mission_dir)
    payload = json.loads(explanation_path.read_text(encoding="utf-8"))

    assert result.status == "refused_by_learning"
    assert payload["decision_type"] == "refusal"
    assert payload["final_resolution"] == "refused"
    assert payload["planner_conflict"] is True

    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Learning decision explanation:" in summary_text


@pytest.mark.asyncio
async def test_learning_bias_only_writes_explanation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    logs_dir = tmp_path / "learning_logs"
    _write_learning_record(
        logs_dir,
        {
            "mission_id": "m-login-1",
            "mission_type": "login",
            "timestamp": "2026-01-13T12:00:00+00:00",
            "confidence_score": 0.9,
            "skills_used": [{"skill_name": "login_form_skill", "success": True, "steps_saved": 2}],
        },
    )

    spec = MissionSpec(instruction="Login demo", execute=True)
    subgoals = [MissionSubgoal(id="sg-login", description="01. Login: dom_presence_check", planner_metadata={})]
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda *_: subgoals)

    worker = _StubWorker()
    executor = MissionExecutor(
        settings={"learning": {"logs_dir": str(logs_dir)}},
        artifacts_root=tmp_path / "artifacts",
        trace_recorder_factory=lambda: ExecutionTraceRecorder(storage_dir=tmp_path / "traces"),
    )

    monkeypatch.setattr(MissionExecutor, "_build_worker", lambda self, spec: worker)
    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", _success_pipeline)
    monkeypatch.setattr(executor, "_compute_learning_score", lambda *_, **__: 0.6)

    result = await executor.run_mission(spec)

    mission_dir = Path(result.artifacts_path)
    summary_path, explanation_path = _mission_result_paths(mission_dir)
    payload = json.loads(explanation_path.read_text(encoding="utf-8"))

    assert result.status == "complete"
    assert payload["decision_type"] == "bias_applied"
    assert payload["final_resolution"] == "bias_only"
    assert payload["triggering_signals"]

    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Learning decision explanation:" in summary_text


@pytest.mark.asyncio
async def test_no_learning_does_not_emit_explanation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = MissionSpec(instruction="Plain mission", execute=True)
    subgoals = [MissionSubgoal(id="sg-1", description="01. Navigate", planner_metadata={})]
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda *_: subgoals)

    executor = MissionExecutor(
        settings={"learning": {"override_threshold": -1.0, "hard_floor": -0.6}},
        artifacts_root=tmp_path / "artifacts",
        trace_recorder_factory=lambda: ExecutionTraceRecorder(storage_dir=tmp_path / "traces"),
    )

    monkeypatch.setattr(MissionExecutor, "_build_worker", lambda self, spec: _StubWorker())
    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", _success_pipeline)
    monkeypatch.setattr(executor, "_compute_learning_score", lambda *_, **__: 0.2)

    result = await executor.run_mission(spec)

    mission_dir = Path(result.artifacts_path)
    summary_path, explanation_path = _mission_result_paths(mission_dir)

    assert result.status == "complete"
    assert not explanation_path.exists()
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Learning decision explanation:" not in summary_text
