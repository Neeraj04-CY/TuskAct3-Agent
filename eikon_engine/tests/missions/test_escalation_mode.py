from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import json

from eikon_engine.missions.mission_executor import MissionExecutor, ESCALATION_STEP_BONUS
from eikon_engine.missions.models import AutonomyBudget, BudgetMonitor
from eikon_engine.runtime.escalation_state import EscalationState

UTC = timezone.utc


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_escalation_emits_artifacts_and_single_use(tmp_path: Path):
    executor = MissionExecutor()
    base_budget = AutonomyBudget(max_steps=5, max_retries=1, max_duration_sec=10, max_risk_score=0.2)
    budget_monitor = BudgetMonitor(base_budget)
    state = EscalationState()

    artifacts = executor._enter_escalation(  # type: ignore[attr-defined]
        mission_dir=tmp_path,
        trace_recorder=None,
        escalation_state=state,
        budget_monitor=budget_monitor,
        base_budget=base_budget,
        expanded_budget=executor._compute_escalation_budget(base_budget),
        window_limits={"time_limit_sec": 5, "step_bonus": ESCALATION_STEP_BONUS},
        limit_detail={"risk_score": 0.5},
    )

    required = {"escalation_request", "escalation_decision", "escalation_window", "escalation_summary"}
    assert required.issubset(set(artifacts.keys()))
    assert state.used is True
    assert state.allowed is False
    assert budget_monitor.budget.max_steps > base_budget.max_steps
    assert Path(artifacts["escalation_request"]).exists()


def test_escalation_window_closes_and_resets_budget(tmp_path: Path):
    executor = MissionExecutor()
    base_budget = AutonomyBudget(max_steps=5, max_retries=1, max_duration_sec=10, max_risk_score=0.2)
    budget_monitor = BudgetMonitor(base_budget)
    state = EscalationState()
    artifacts = executor._enter_escalation(  # type: ignore[attr-defined]
        mission_dir=tmp_path,
        trace_recorder=None,
        escalation_state=state,
        budget_monitor=budget_monitor,
        base_budget=base_budget,
        expanded_budget=executor._compute_escalation_budget(base_budget),
        window_limits={"time_limit_sec": 0, "step_bonus": ESCALATION_STEP_BONUS},
        limit_detail={"risk_score": 0.5},
    )
    # Simulate time elapsed
    state.started_at = (datetime.now(UTC) - timedelta(seconds=2)).isoformat()
    executor._check_escalation_window(  # type: ignore[attr-defined]
        mission_dir=tmp_path,
        trace_recorder=None,
        escalation_state=state,
        budget_monitor=budget_monitor,
        base_budget=base_budget,
        artifacts=artifacts,
    )
    assert state.ended_at is not None
    assert budget_monitor.budget.max_steps == base_budget.max_steps
    window_payload = _read_json(Path(artifacts["escalation_window"]))
    assert window_payload.get("ended_at") is not None


def test_escalation_state_persisted_in_checkpoint(tmp_path: Path):
    from eikon_engine.runtime.resume_checkpoint import ResumeCheckpoint

    state = EscalationState(used=True, started_at=datetime.now(UTC).isoformat(), expanded_budget={"max_steps": 50})
    checkpoint = ResumeCheckpoint(
        mission_id="m1",
        halted_subgoal_id="sg1",
        halted_reason="autonomy_budget_exceeded",
        page_url=None,
        page_intent=None,
        completed_subgoals=[],
        pending_subgoals=[],
        skills_used=[],
        capability_state={},
        learning_bias_snapshot={},
        trace_path="trace.json",
        timestamp_utc=datetime.now(UTC).isoformat(),
        escalation_state=state.to_dict(),
        mission_instruction="goal",
        artifacts_path=str(tmp_path),
    )
    path = checkpoint.save(tmp_path / "resume_checkpoint.json")
    loaded = ResumeCheckpoint.load(path)
    assert loaded.escalation_state.get("used") is True
    assert loaded.escalation_state.get("expanded_budget").get("max_steps") == 50
