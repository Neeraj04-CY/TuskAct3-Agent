from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from eikon_engine.learning.recorder import LearningRecorder
from eikon_engine.missions.mission_executor import MissionExecutor
from eikon_engine.missions.mission_schema import MissionSpec, MissionSubgoal
from eikon_engine.judgment.evaluator import JudgmentEvaluator
from eikon_engine.trace.reader import read_trace
from eikon_engine.trace.recorder import ExecutionTraceRecorder
from eikon_engine.replay import ReplayConfig, ReplayEngine
from eikon_engine.trace.models import ActionTrace, ArtifactRecord, ExecutionTrace, SubgoalTrace


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
        raise AssertionError("pipeline should not execute when judgment halts")

    async def run_skill(self, *_: object, **__: object) -> dict:  # pragma: no cover
        return {"status": "success"}


@pytest.mark.asyncio
async def test_judgment_autonomous_halt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec = MissionSpec(instruction="Explore unless authority required", execute=False)
    subgoals = [
        MissionSubgoal(
            id="sg-judgment",
            description="Attempt action requiring impersonation",
            planner_metadata={
                "capability_requirements": [
                    {
                        "capability_id": "identity.assume",
                        "required": True,
                        "confidence": 0.5,
                        "reason": "needs identity",
                    }
                ]
            },
        )
    ]
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda *_: subgoals)

    trace_dir = tmp_path / "traces"
    executor = MissionExecutor(
        settings={"planner": {}, "approval": {"timeout_secs": 5}, "capability_enforcement": {"threshold": 0.8, "critical": 0.5}},
        artifacts_root=tmp_path,
        trace_recorder_factory=lambda: ExecutionTraceRecorder(storage_dir=trace_dir),
    )
    monkeypatch.setattr(MissionExecutor, "_build_worker", lambda self, spec: _StubWorker())

    result = await executor.run_mission(spec)

    assert result.status == "halted"
    assert result.summary.get("reason") == "judgment_refusal"

    decision_path = Path(result.summary.get("judgment_decision_path"))
    explanation_path = Path(result.summary.get("decision_explanation_path"))
    assert decision_path.exists()
    assert explanation_path.exists()

    trace = read_trace(Path(result.summary["execution_trace"]))
    assert "Agent halted execution due to autonomous judgment." in (trace.warnings or [])
    summary_text = Path(result.summary["execution_trace_summary"]).read_text(encoding="utf-8")
    assert "Agent halted execution due to autonomous judgment." in summary_text


@pytest.mark.asyncio
async def test_judgment_requests_approval_on_low_confidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec = MissionSpec(instruction="Explore with caution", execute=False)
    subgoals = [MissionSubgoal(id="sg-approval", description="Collect info", planner_metadata={})]
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda *_: subgoals)

    trace_dir = tmp_path / "traces"
    executor = MissionExecutor(
        settings={
            "planner": {},
            "approval": {"timeout_secs": 5, "require_approval": False},
            "capability_enforcement": {"threshold": 0.8, "critical": 0.5},
            "learning": {"enable_bias": True},
        },
        artifacts_root=tmp_path,
        trace_recorder_factory=lambda: ExecutionTraceRecorder(storage_dir=trace_dir),
    )

    low_confidence_bias = type(
        "Bias",
        (),
        {"confidence": 0.1, "preferred_skills": [], "signals": [], "as_metadata": lambda self=None: {}},
    )()  # simple stub
    monkeypatch.setattr(MissionExecutor, "_build_worker", lambda self, spec: _StubWorker())
    monkeypatch.setattr(MissionExecutor, "_resolve_learning_bias", lambda self, spec: low_confidence_bias)

    result = await executor.run_mission(spec)

    assert result.status == "ask_human"
    assert result.summary.get("reason") == "judgment_request_approval"
    approval_paths = result.summary.get("approval_requests") or []
    assert approval_paths
    assert Path(approval_paths[0]).exists()


@pytest.mark.asyncio
async def test_replay_handles_halted_trace(tmp_path: Path) -> None:
    ts = datetime.now(timezone.utc)
    action = ActionTrace(
        id="action_1",
        sequence=1,
        started_at=ts,
        ended_at=ts,
        duration_ms=1.0,
        action_type="navigate",
        selector=None,
        target="https://example.com",
        input_data=None,
        status="ok",
        metadata={},
    )
    subgoal = SubgoalTrace(
        id="sg0",
        subgoal_id="sg0",
        description="halted navigation",
        attempt_number=1,
        started_at=ts,
        status="complete",
        actions_taken=[action],
    )
    trace = ExecutionTrace(
        id="trace_halted",
        mission_id="mission_halted",
        mission_text="halt demo",
        started_at=ts,
        status="halted",
        subgoal_traces=[subgoal],
        artifacts=[ArtifactRecord(id="art1", name="subgoal_01", path=str(tmp_path / "sg"), started_at=ts)],
        warnings=["Agent halted execution due to autonomous judgment."],
    )

    engine = ReplayEngine(config=ReplayConfig(output_root=tmp_path / "replay", worker_factory=lambda logger, headless: _StubWorker()))
    summary = await engine.replay_trace(trace)

    assert summary.status in {"success", "failed"}  # no crash
    assert summary.output_dir.exists()


def test_learning_log_marks_halted(tmp_path: Path) -> None:
    mission_dir = tmp_path / "mission"
    mission_dir.mkdir(parents=True, exist_ok=True)
    mission_result = {
        "mission_id": "mission_halted",
        "status": "halted",
        "artifacts_path": str(mission_dir),
        "mission": "halt demo",
    }
    mission_result_path = mission_dir / "mission_result.json"
    mission_result_path.write_text(json.dumps(mission_result), encoding="utf-8")

    trace_path = mission_dir / "trace.json"
    trace_path.write_text(json.dumps({"id": "trace_halted", "status": "halted"}), encoding="utf-8")

    recorder = LearningRecorder(output_dir=tmp_path / "learning_logs")
    recorder.record(mission_result_path=mission_result_path, trace_path=trace_path, mission_instruction="halt demo")

    log_path = tmp_path / "learning_logs" / "mission_halted.json"
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload.get("outcome") == "halted"


def test_judgment_handles_dict_page_intent() -> None:
    evaluator = JudgmentEvaluator()
    spec = MissionSpec(instruction="Check login gate", execute=False)
    subgoal = MissionSubgoal(id="sg-intent", description="Visit portal", planner_metadata={})

    decision = evaluator.evaluate(
        mission_spec=spec,
        subgoal=subgoal,
        capability_requirements=[],
        safety_contract=None,
        learning_bias=None,
        predicted_actions=[],
        page_intent={"intent": "login_portal", "confidence": 0.9},
    )

    assert decision.decision == "request_approval"
    assert "page_intent_login" in decision.risk_factors
