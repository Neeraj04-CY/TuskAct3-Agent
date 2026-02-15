from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .models import LearningFailure, LearningSkillUsage


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _count_strategy_violations(trace: Any) -> int:
    failures = _get(trace, "failures", []) or []
    count = 0
    for entry in failures:
        failure_type = _get(entry, "failure_type") or _get(entry, "type")
        message = _get(entry, "message", "") or ""
        text = f"{failure_type} {message}".lower()
        if "strategy_violation" in text:
            count += 1
    return count


def _count_replans(trace: Any) -> int:
    evaluation = _get(trace, "evaluation", {}) or {}
    replans = 0
    if isinstance(evaluation, dict):
        replans = int(evaluation.get("replans", 0) or 0)
    subgoals = _get(trace, "subgoal_traces", []) or []
    if subgoals:
        replans = max(
            replans,
            sum(1 for subgoal in subgoals if _get(subgoal, "attempt_number", 1) and _get(subgoal, "attempt_number", 1) > 1),
        )
    return replans


def _gather_skill_usage(trace: Any, skill_summary: Optional[Iterable[Dict[str, Any]]]) -> List[LearningSkillUsage]:
    if skill_summary:
        skills = []
        for entry in skill_summary:
            name = str(entry.get("skill_name") or entry.get("name") or "unknown")
            status = str(entry.get("status") or "").lower()
            success = status == "success"
            steps_saved = int(entry.get("steps_saved", 0) or 0)
            skills.append(LearningSkillUsage(skill_name=name, success=success, steps_saved=steps_saved))
        return skills
    skills = []
    for usage in _get(trace, "skills_used", []) or []:
        name = _get(usage, "name", "unknown")
        status = _get(usage, "status", "")
        success = str(status).lower() == "success"
        steps_saved = 0
        metadata = _get(usage, "metadata", {}) or {}
        if isinstance(metadata, dict):
            maybe_steps = metadata.get("steps_saved")
            if isinstance(maybe_steps, (int, float)):
                steps_saved = int(maybe_steps)
        skills.append(LearningSkillUsage(skill_name=str(name), success=success, steps_saved=steps_saved))
    return skills


def _gather_failures(trace: Any) -> List[LearningFailure]:
    failures = []
    for entry in _get(trace, "failures", []) or []:
        step = _get(entry, "subgoal_id") or _get(entry, "id", "unknown_step")
        reason = _get(entry, "message", "") or "unknown"
        failures.append(LearningFailure(step=str(step), reason=str(reason)))
    return failures


def score_learning(
    *,
    mission_result: Any,
    trace: Any,
    skill_summary: Optional[Iterable[Dict[str, Any]]],
    runtime_error: Any | None = None,
    artifacts_exist: bool = True,
    force_outcome_failure: bool = False,
) -> Dict[str, Any]:
    skills = _gather_skill_usage(trace, skill_summary)
    failures = _gather_failures(trace)
    if runtime_error:
        failures.append(LearningFailure(step="runtime", reason=str(runtime_error)))
    if not artifacts_exist:
        failures.append(LearningFailure(step="artifacts", reason="missing"))

    positives = 0
    negatives = 0

    # Skill success -> +1
    positives += sum(1 for entry in skills if entry.success)

    # Skill prevented retries (steps_saved > 0) -> +1
    positives += sum(1 for entry in skills if entry.steps_saved > 0)

    # Mission success -> +1
    outcome_status = str(_get(mission_result, "status", "")).lower()
    if outcome_status == "complete":
        positives += 1

    # Strategy violation -> -2
    negatives += 2 * _count_strategy_violations(trace)

    # Replan loop -> -1
    negatives += _count_replans(trace)

    raw = positives - negatives
    span = max(positives + negatives, 1)
    normalized = (raw + span) / (2 * span)
    confidence = max(0.0, min(1.0, normalized))

    has_skill_failure = any(not entry.success for entry in skills)
    has_failures = bool(failures)
    bad_runtime = runtime_error is not None
    bad_artifacts = not artifacts_exist

    bad_conditions = has_skill_failure or has_failures or bad_runtime or bad_artifacts or force_outcome_failure
    if outcome_status not in {"complete", "halted"}:
        bad_conditions = True

    if bad_conditions:
        confidence = min(confidence, 0.49)

    if force_outcome_failure or bad_runtime or has_skill_failure or bad_artifacts:
        outcome = "failure"
    elif outcome_status == "halted":
        outcome = "halted"
    elif outcome_status == "complete" and not has_failures:
        outcome = "success"
    else:
        outcome = "failure"

    try:
        summary = mission_result.get("summary") if isinstance(mission_result, dict) else None
        if summary and isinstance(summary, dict):
            if (summary.get("escalation_state") or {}).get("used"):
                confidence = max(confidence - 0.1, 0.0)
    except Exception:
        pass

    return {
        "skills": skills,
        "failures": failures,
        "confidence": confidence,
        "outcome": outcome,
    }


__all__ = ["score_learning"]
