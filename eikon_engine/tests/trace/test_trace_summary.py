from __future__ import annotations

from datetime import datetime, timezone

from eikon_engine.trace.models import ExecutionTrace, PageIntentRecord, SkillUsage, SubgoalSkipRecord, SubgoalTrace
from eikon_engine.trace.summary import build_trace_summary

UTC = timezone.utc


def _ts() -> datetime:
    return datetime(2025, 1, 1, tzinfo=UTC)


def test_trace_summary_storyline_highlights_intent_flow() -> None:
    started = _ts()
    trace = ExecutionTrace(
        id="trace-123",
        mission_id="mission-123",
        mission_text="Open https://www.ycombinator.com/companies",
        started_at=started,
        ended_at=started,
        duration_ms=1000.0,
        status="complete",
        subgoal_traces=[
            SubgoalTrace(
                id="sg0_attempt1",
                subgoal_id="sg0",
                description="00. navigation: navigate to https://www.ycombinator.com/companies",
                attempt_number=1,
                started_at=started,
                ended_at=started,
                duration_ms=1000.0,
                status="complete",
            )
        ],
        skills_used=[
            SkillUsage(
                id="skill-1",
                name="listing_extraction_skill",
                status="success",
                metadata={
                    "result": {
                        "items_found": 2,
                        "result": {"company_name": "Airbnb"},
                    }
                },
            )
        ],
        page_intents=[
            PageIntentRecord(
                id="intent-1",
                intent="LISTING_PAGE",
                strategy="listing_extraction",
                confidence=0.9,
                signals={},
                step_id="dom_probe",
                decided_at=started,
            )
        ],
        skipped_subgoals=[
            SubgoalSkipRecord(
                id="skip-1",
                subgoal_id="sg1",
                description="01. navigation: navigate",
                reason="page_intent_known",
                page_intent="LISTING_PAGE",
                decided_at=started,
            )
        ],
    )

    summary = build_trace_summary(trace)

    checkpoints = [
        "Navigation completed",
        "Page intent detected: LISTING_PAGE",
        "Navigation/form subgoals skipped",
        "Listing extraction skill executed",
        "Mission completed successfully.",
    ]
    positions = [summary.index(text) for text in checkpoints]
    assert positions == sorted(positions)
    assert "Example.com references: none" in summary
    assert "Planner replans triggered: none" in summary
    assert "Unknown intents observed: none" in summary


def test_trace_summary_flags_example_domain_mentions() -> None:
    started = _ts()
    trace = ExecutionTrace(
        id="trace-456",
        mission_id="mission-456",
        mission_text="Navigate to https://example.com",
        started_at=started,
        ended_at=started,
        duration_ms=500.0,
        status="complete",
    )

    summary = build_trace_summary(trace)

    assert "Example.com references detected" in summary