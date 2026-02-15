from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from eikon_engine.replay import ReplayConfig, ReplayEngine
from eikon_engine.trace.models import (
    ActionTrace,
    ApprovalRequestRecord,
    ApprovalResolutionRecord,
    ArtifactRecord,
    ExecutionTrace,
    SkillUsage,
    SubgoalTrace,
)

UTC = timezone.utc


class _StubWorker:
    def __init__(self, *, steps: List[Dict[str, Any]], skill_payload: Dict[str, Any] | None = None) -> None:
        self._steps = steps
        self._skill_payload = skill_payload or {"status": "success", "result": {"company_name": "Airbnb"}}
        self.executed: List[Dict[str, Any]] = []
        self.skill_calls: List[Dict[str, Any]] = []

    async def execute(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        self.executed.append(metadata)
        return {
            "steps": self._steps,
            "dom_snapshot": "<html></html>",
        }

    async def run_skill(self, skill_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.skill_calls.append({"skill": skill_name, "context": context})
        return dict(self._skill_payload)

    async def shutdown(self) -> None:  # pragma: no cover - nothing to clean up in stub
        return None


@pytest.mark.asyncio
async def test_replay_engine_replays_actions_and_skill(tmp_path: Path) -> None:
    subgoal_artifacts = tmp_path / "original_subgoal"
    (subgoal_artifacts / "step_001").mkdir(parents=True)
    (subgoal_artifacts / "step_001" / "dom.html").write_text("<html>artifact</html>", encoding="utf-8")

    action_started = datetime.now(UTC)
    actions = [
        ActionTrace(
            id="sg_action_001",
            sequence=1,
            started_at=action_started,
            ended_at=action_started,
            duration_ms=10.0,
            action_type="navigate",
            selector=None,
            target="https://example.com",
            input_data=None,
            status="ok",
            metadata={},
        ),
        ActionTrace(
            id="sg_action_002",
            sequence=2,
            started_at=action_started,
            ended_at=action_started,
            duration_ms=5.0,
            action_type="screenshot",
            selector=None,
            target=None,
            input_data=None,
            status="ok",
            metadata={},
        ),
    ]
    subgoal = SubgoalTrace(
        id="sg_handle",
        subgoal_id="sg0",
        description="Replay test subgoal",
        attempt_number=1,
        started_at=action_started,
        status="complete",
        actions_taken=actions,
        skill_used="listing_extraction_skill",
    )
    skill_usage = SkillUsage(
        id="skill_001",
        name="listing_extraction_skill",
        status="success",
        subgoal_id="sg0",
        metadata={
            "result": {"status": "success", "result": {"company_name": "Airbnb"}},
        },
    )
    trace = ExecutionTrace(
        id="trace_01",
        mission_id="mission_01",
        mission_text="test",
        started_at=action_started,
        status="complete",
        subgoal_traces=[subgoal],
        skills_used=[skill_usage],
        artifacts=[
            ArtifactRecord(id="artifact_1", name="subgoal_01", path=str(subgoal_artifacts), started_at=action_started),
        ],
    )

    stub_worker = _StubWorker(
        steps=[
            {"action": "navigate", "status": "ok"},
            {"action": "screenshot", "status": "ok"},
        ]
    )

    def worker_factory(logger, headless):  # type: ignore[unused-argument]
        return stub_worker

    config = ReplayConfig(output_root=tmp_path / "replay", worker_factory=worker_factory)
    engine = ReplayEngine(config=config)

    summary = await engine.replay_trace(trace)

    assert summary.status == "success"
    assert summary.actions_replayed == 2
    assert summary.skills_replayed == 1
    assert summary.skill_details and summary.skill_details[0]["skill"] == "listing_extraction_skill"
    assert (summary.output_dir / "replay_summary.txt").exists()


@pytest.mark.asyncio
async def test_replay_engine_detects_step_divergence(tmp_path: Path) -> None:
    action_started = datetime.now(UTC)
    action = ActionTrace(
        id="sg_action_001",
        sequence=1,
        started_at=action_started,
        ended_at=action_started,
        duration_ms=10.0,
        action_type="navigate",
        selector=None,
        target="https://example.com",
        input_data=None,
        status="ok",
        metadata={},
    )
    subgoal = SubgoalTrace(
        id="sg_handle",
        subgoal_id="sg0",
        description="Replay failure subgoal",
        attempt_number=1,
        started_at=action_started,
        status="complete",
        actions_taken=[action],
    )
    trace = ExecutionTrace(
        id="trace_fail",
        mission_id="mission_fail",
        mission_text="test",
        started_at=action_started,
        status="complete",
        subgoal_traces=[subgoal],
        artifacts=[],
    )

    def worker_factory(logger, headless):  # type: ignore[unused-argument]
        return _StubWorker(steps=[{"action": "navigate", "status": "error"}])

    engine = ReplayEngine(config=ReplayConfig(output_root=tmp_path / "replay_fail", worker_factory=worker_factory))

    summary = await engine.replay_trace(trace)

    assert summary.status == "failed"
    assert summary.divergence is not None
    assert "observed" in summary.divergence


@pytest.mark.asyncio
async def test_replay_engine_ignores_approval_metadata(tmp_path: Path) -> None:
    action_started = datetime.now(UTC)
    action = ActionTrace(
        id="sg_action_001",
        sequence=1,
        started_at=action_started,
        ended_at=action_started,
        duration_ms=5.0,
        action_type="navigate",
        selector=None,
        target="https://example.com",
        input_data=None,
        status="ok",
        metadata={},
    )
    subgoal = SubgoalTrace(
        id="sg_handle",
        subgoal_id="sg0",
        description="Replay approval subgoal",
        attempt_number=1,
        started_at=action_started,
        status="complete",
        actions_taken=[action],
    )
    approval_request = ApprovalRequestRecord(
        id="approval_req_001",
        approval_id="apr_1",
        subgoal_id="sg0",
        reason="replay",
        risk_level="medium",
        requested_action={"type": "subgoal", "name": "Replay approval subgoal", "target": "example"},
        created_at=action_started,
        expires_at=action_started,
    )
    approval_resolution = ApprovalResolutionRecord(
        id="approval_res_001",
        approval_id="apr_1",
        subgoal_id="sg0",
        state="approved",
        resolved_at=action_started,
        resolved_by="test",
        reason="approved_for_replay",
        external=True,
    )
    trace = ExecutionTrace(
        id="trace_approval",
        mission_id="mission_with_approval",
        mission_text="test",
        started_at=action_started,
        status="complete",
        subgoal_traces=[subgoal],
        artifacts=[],
        approvals_requested=[approval_request],
        approvals_resolved=[approval_resolution],
    )

    def worker_factory(logger, headless):  # type: ignore[unused-argument]
        return _StubWorker(steps=[{"action": "navigate", "status": "ok"}])

    engine = ReplayEngine(config=ReplayConfig(output_root=tmp_path / "replay_approval", worker_factory=worker_factory))

    summary = await engine.replay_trace(trace)

    assert summary.status == "success"
    assert summary.actions_replayed == 1
    assert summary.subgoals_replayed == 1
    assert summary.divergence is None
    assert (summary.output_dir / "replay_summary.txt").exists()