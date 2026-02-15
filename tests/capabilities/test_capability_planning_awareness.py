from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pytest

from eikon_engine.capabilities.inference import build_plan_capability_report, plan_capability_report_for_tasks
from eikon_engine.capabilities.models import CapabilityRequirement
from eikon_engine.capabilities.registry import CAPABILITY_REGISTRY
from eikon_engine.missions.mission_executor import MissionExecutor
from eikon_engine.missions.mission_schema import MissionSpec, MissionSubgoal
from eikon_engine.trace.recorder import ExecutionTraceRecorder
from eikon_engine.trace.summary import build_trace_summary

UTC = timezone.utc


def test_plan_capability_report_inference_low_risk() -> None:
    tasks = [
        {"id": "task_1", "bucket": "navigation"},
        {"id": "task_2", "bucket": "listing_extraction"},
    ]
    report, requirements = plan_capability_report_for_tasks(tasks, registry=CAPABILITY_REGISTRY)
    assert report.risk_level == "low"
    assert requirements["task_1"][0].capability_id == "web_navigation"
    assert requirements["task_2"][0].capability_id == "data_extraction"


def test_capability_report_risk_levels_required_vs_optional() -> None:
    optional_req = CapabilityRequirement(capability_id="optional.cap", required=False, confidence=0.4, reason="optional")
    required_req = CapabilityRequirement(capability_id="required.cap", required=True, confidence=0.8, reason="required")
    medium_report = build_plan_capability_report([optional_req], registry={})
    assert medium_report.risk_level == "medium"
    high_report = build_plan_capability_report([required_req], registry={})
    assert high_report.risk_level == "high"


def test_capability_report_persisted_in_trace_summary(tmp_path: Path) -> None:
    mission_spec = MissionSpec(instruction="Demo", execute=False)
    mission_dir = tmp_path / "mission"
    mission_dir.mkdir()
    recorder = ExecutionTraceRecorder(storage_dir=tmp_path)
    started_at = datetime.now(UTC)
    recorder.start(mission_spec=mission_spec, mission_dir=mission_dir, started_at=started_at)
    capability_payload: Dict[str, object] = {
        "required": [
            {"capability_id": "web_navigation", "required": True, "confidence": 0.9, "reason": "bucket:navigation"}
        ],
        "missing": [],
        "optional": [],
        "risk_level": "low",
    }
    recorder.record_capability_report(capability_payload)
    recorder.finalize(status="complete", ended_at=started_at)
    trace_path = recorder.persist()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["capability_report"]["risk_level"] == "low"
    summary_text = build_trace_summary(recorder.trace)  # type: ignore[arg-type]
    assert "Plan capability risk" in summary_text


@pytest.mark.asyncio
async def test_mission_execution_not_blocked_when_capabilities_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = MissionSpec(instruction="Do something", execute=False)
    subgoals = [
        MissionSubgoal(
            id="sg1",
            description="Navigate somewhere",
            planner_metadata={
                "capability_requirements": [
                    {"capability_id": "unknown.capability", "required": True, "confidence": 0.9, "reason": "test"}
                ]
            },
        )
    ]
    monkeypatch.setattr(
        "eikon_engine.missions.mission_planner.plan_mission",
        lambda _spec, settings=None: subgoals,
    )

    async def fake_run_pipeline(self, **_: object) -> dict:  # noqa: ARG001
        return {
            "completion": {"complete": True, "reason": "ok"},
            "artifacts": {"base_dir": str(tmp_path / "subgoal")},
        }

    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", fake_run_pipeline)

    executor = MissionExecutor(
        settings={"planner": {}, "logging": {"artifact_root": str(tmp_path)}},
        artifacts_root=tmp_path,
    )
    result = await executor.run_mission(spec)
    assert result.status == "complete"
    capability_report = result.summary.get("capability_report", {})
    assert capability_report.get("risk_level") == "high"
    report_path = result.summary.get("capability_report_path")
    assert report_path is not None and Path(report_path).exists()
