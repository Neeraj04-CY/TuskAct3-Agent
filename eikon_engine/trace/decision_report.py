from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from .models import ExecutionTrace, FailureRecord, PageIntentRecord, SkillUsage, SubgoalTrace


def build_decision_report(trace: ExecutionTrace) -> Dict[str, Any]:
    """Aggregate decision-attribution, confidence, and failure metadata."""

    confidence = _confidence_summary(trace.page_intents)
    failures = _failure_summary(trace.failures)
    decisions = _decision_events(trace)
    risk_flags = _risk_flags(confidence, trace, failures)
    return {
        "mission_id": trace.mission_id,
        "trace_id": trace.id,
        "status": trace.status,
        "confidence": confidence,
        "risk_flags": risk_flags,
        "decisions": decisions,
        "failures": failures,
    }


def write_decision_report(
    trace: ExecutionTrace,
    trace_file: Path,
    *,
    output_dir: Path | None = None,
) -> Path:
    """Persist a machine-readable decision report next to the trace file."""

    report = build_decision_report(trace)
    target_dir = output_dir if output_dir else trace_file.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "trace_decisions.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def _confidence_summary(page_intents: List[PageIntentRecord]) -> Dict[str, Any]:
    scores = [record.confidence for record in page_intents if record.confidence is not None]
    if not scores:
        return {"samples": 0, "average": None, "min": None, "max": None}
    avg = round(mean(scores), 3)
    minimum = round(min(scores), 3)
    maximum = round(max(scores), 3)
    return {
        "samples": len(scores),
        "average": avg,
        "min": minimum,
        "max": maximum,
    }


def _failure_summary(failures: List[FailureRecord]) -> Dict[str, Any]:
    if not failures:
        return {"total": 0, "entries": [], "categories": []}
    entries: List[Dict[str, Any]] = []
    categories: List[str] = []
    for failure in failures:
        category = _classify_failure(failure)
        categories.append(category)
        entries.append(
            {
                "id": failure.id,
                "type": failure.failure_type,
                "message": failure.message,
                "retryable": failure.retryable,
                "subgoal_id": failure.subgoal_id,
                "category": category,
            }
        )
    return {
        "total": len(entries),
        "entries": entries,
        "categories": sorted(set(categories)),
    }


def _decision_events(trace: ExecutionTrace) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for record in trace.page_intents:
        events.append(
            {
                "type": "page_intent",
                "intent": record.intent,
                "confidence": record.confidence,
                "strategy": record.strategy,
                "signals": record.signals,
                "step_id": record.step_id,
                "decided_at": record.decided_at.isoformat() if record.decided_at else None,
            }
        )
    for usage in trace.skills_used:
        events.append(
            {
                "type": "skill_usage",
                "skill": usage.name,
                "status": usage.status,
                "subgoal_id": usage.subgoal_id,
                "metadata": usage.metadata,
            }
        )
    for subgoal in trace.subgoal_traces:
        events.append(
            {
                "type": "subgoal",
                "subgoal_id": subgoal.subgoal_id,
                "description": subgoal.description,
                "status": subgoal.status,
                "attempts": subgoal.attempt_number,
            }
        )
    return events


def _risk_flags(
    confidence: Dict[str, Any],
    trace: ExecutionTrace,
    failures: Dict[str, Any],
) -> List[str]:
    flags: List[str] = []
    samples = confidence.get("samples") or 0
    avg = confidence.get("average")
    if not samples:
        flags.append("unknown_confidence")
    elif isinstance(avg, (int, float)):
        if avg < 0.5:
            flags.append("very_low_confidence")
        elif avg < 0.75:
            flags.append("low_confidence")
    if any((subgoal.attempt_number or 1) > 1 for subgoal in trace.subgoal_traces):
        flags.append("retries_detected")
    if failures.get("total"):
        flags.append("failures_recorded")
    if trace.incomplete:
        flags.append("trace_incomplete")
    return sorted(set(flags))


def _classify_failure(failure: FailureRecord) -> str:
    failure_type = (failure.failure_type or "").lower()
    message = (failure.message or "").lower()
    if "timeout" in failure_type or "timeout" in message:
        return "timeout"
    if "planner" in failure_type:
        return "planner"
    if "strategy_violation" in failure_type:
        return "strategy_violation"
    if "subgoal" in failure_type:
        return "subgoal_execution"
    if failure.retryable:
        return "retryable_failure"
    return "unknown"


__all__ = ["build_decision_report", "write_decision_report"]
