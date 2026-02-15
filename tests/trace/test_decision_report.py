from __future__ import annotations

from datetime import datetime, timezone

from eikon_engine.trace.decision_report import build_decision_report
from eikon_engine.trace.models import (
    ExecutionTrace,
    FailureRecord,
    PageIntentRecord,
    SkillUsage,
    SubgoalTrace,
)

UTC = timezone.utc


def test_decision_report_builds_risk_flags() -> None:
    now = datetime.now(UTC)
    subgoal = SubgoalTrace(
        id="sg_handle",
        subgoal_id="sg0",
        description="Sample",
        attempt_number=2,
        started_at=now,
        status="failed",
    )
    page_intent = PageIntentRecord(
        id="intent_001",
        intent="listing_page",
        strategy="listing_extraction",
        confidence=0.42,
        signals={"card_count": 10},
        decided_at=now,
    )
    failure = FailureRecord(
        id="failure_001",
        failure_type="mission_timeout",
        message="deadline exceeded",
        subgoal_id="sg0",
        retryable=False,
        started_at=now,
    )
    trace = ExecutionTrace(
        id="trace_report",
        mission_id="mission_report",
        mission_text="test",
        started_at=now,
        status="failed",
        subgoal_traces=[subgoal],
        page_intents=[page_intent],
        failures=[failure],
        skills_used=[SkillUsage(id="skill_001", name="listing_extraction_skill", status="error", subgoal_id="sg0")],
        incomplete=True,
    )

    report = build_decision_report(trace)

    assert report["confidence"]["average"] == round(0.42, 3)
    assert "failures_recorded" in report["risk_flags"]
    assert any(event["type"] == "page_intent" for event in report["decisions"])
    assert report["failures"]["total"] == 1