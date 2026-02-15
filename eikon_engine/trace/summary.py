from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

from .models import ExecutionTrace, PageIntentRecord, SkillUsage, SubgoalSkipRecord, SubgoalTrace


def _format_duration(duration_ms: float | None) -> str:
    if duration_ms is None:
        return "unknown"
    return f"{duration_ms / 1000:.1f}s"


def _joined(values: Iterable[str]) -> str:
    unique = sorted({value for value in values if value})
    return ", ".join(unique) if unique else "none"


def build_trace_summary(trace: ExecutionTrace) -> str:
    total_subgoals = len(trace.subgoal_traces)
    succeeded = sum(1 for subgoal in trace.subgoal_traces if subgoal.status == "complete")
    retried = sum(1 for subgoal in trace.subgoal_traces if subgoal.attempt_number > 1)
    actions_executed = sum(len(subgoal.actions_taken) for subgoal in trace.subgoal_traces)
    skills_used = _joined(usage.name for usage in trace.skills_used)
    failures_recovered = sum(1 for failure in trace.failures if failure.retryable)
    lines = [
        f"Mission {trace.status} in {_format_duration(trace.duration_ms)}",
        f"Subgoals: {total_subgoals} ({succeeded} succeeded, {retried} retried)",
        f"Actions executed: {actions_executed}",
        f"Skills used: {skills_used}",
        f"Failures recovered: {failures_recovered}",
    ]
    storyline = _build_storyline(trace)
    if storyline:
        lines.append("")
        lines.append("Storyline:")
        lines.extend(storyline)
    guardrails = _build_guardrail_lines(trace)
    if guardrails:
        lines.append("")
        lines.append("Guards:")
        lines.extend(guardrails)
    lifecycle = _lifecycle_lines(trace)
    if lifecycle:
        lines.append("")
        lines.append("Lifecycle:")
        lines.extend(lifecycle)
    capability_report = _capability_report_lines(trace)
    if capability_report:
        lines.append("")
        lines.extend(capability_report)
    enforcement = _capability_enforcement_lines(trace)
    if enforcement:
        lines.append("")
        lines.extend(enforcement)
    approvals = _approval_lines(trace)
    if approvals:
        lines.append("")
        lines.extend(approvals)
    capabilities = _capabilities_lines(trace)
    if capabilities:
        lines.append("")
        lines.extend(capabilities)
    explanation = _learning_explanation_line(trace)
    if explanation:
        lines.append("")
        lines.append(explanation)
    if trace.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in trace.warnings:
            lines.append(f"- {warning}")
    if trace.incomplete:
        lines.append("")
        lines.append("Trace marked incomplete; inspect warnings section.")
    lines.append("")
    lines.append(f"Final status: {trace.status}")
    return "\n".join(line for line in lines if line is not None)


def write_trace_summary(trace: ExecutionTrace, trace_file: Path) -> Path:
    summary_text = build_trace_summary(trace) + "\n"
    summary_path = trace_file.with_name("trace_summary.txt")
    summary_path.write_text(summary_text, encoding="utf-8")
    return summary_path


__all__ = ["build_trace_summary", "write_trace_summary"]


def _build_storyline(trace: ExecutionTrace) -> List[str]:
    nav_line = _navigation_line(trace.subgoal_traces)
    intent_line = _intent_line(trace.page_intents)
    skip_line = _skip_line(trace.skipped_subgoals)
    skill_line = _skill_line(trace.skills_used)
    completion_line = _completion_line(trace.status)
    storyline = [nav_line, intent_line, skip_line, skill_line, completion_line]
    return [line for line in storyline if line]


def _navigation_line(subgoal_traces: List[SubgoalTrace]) -> Optional[str]:
    for subgoal in subgoal_traces:
        if subgoal.status != "complete":
            continue
        description = subgoal.description or "navigation"
        if "navigation" not in description.lower():
            continue
        url = _extract_url(description)
        suffix = f" ({url})" if url else ""
        return f"Navigation completed: {description}{suffix}"
    if subgoal_traces:
        description = subgoal_traces[0].description or "navigation"
        url = _extract_url(description)
        suffix = f" ({url})" if url else ""
        return f"Navigation completed: {description}{suffix}"
    return None


def _intent_line(page_intents: List[PageIntentRecord]) -> Optional[str]:
    if not page_intents:
        return "Page intent detected: none recorded"
    intent = page_intents[-1].intent.upper()
    return f"Page intent detected: {intent}"


def _skip_line(skipped: List[SubgoalSkipRecord]) -> Optional[str]:
    if not skipped:
        return "Navigation/form subgoals skipped: none"
    descriptions = ", ".join(record.description for record in skipped if record.description)
    reasons = _joined(record.reason for record in skipped)
    intents = _joined((record.page_intent or "").upper() for record in skipped if record.page_intent)
    detail_bits = [f"reason {reasons}" if reasons else "", f"intent {intents}" if intents else ""]
    details = ", ".join(bit for bit in detail_bits if bit)
    suffix = f" ({details})" if details else ""
    return f"Navigation/form subgoals skipped ({len(skipped)}): {descriptions}{suffix}"


def _skill_line(skills: List[SkillUsage]) -> Optional[str]:
    if not skills:
        return "Listing extraction skill executed: none"
    listing_skill = next((skill for skill in skills if skill.name == "listing_extraction_skill"), None)
    if not listing_skill:
        listing_skill = skills[0]
    metadata = listing_skill.metadata or {}
    items_found = metadata.get("result", {}).get("items_found")
    company = metadata.get("result", {}).get("result", {}).get("company_name") or metadata.get("result", {}).get("result", {}).get("name")
    detail_parts = []
    if items_found is not None:
        detail_parts.append(f"items_found={items_found}")
    if company:
        detail_parts.append(f"company={company}")
    suffix = f" ({', '.join(detail_parts)})" if detail_parts else ""
    return f"Listing extraction skill executed: {listing_skill.status}{suffix}"


def _completion_line(status: str) -> str:
    if status == "complete":
        return "Mission completed successfully."
    return f"Mission ended with status: {status.upper()}"


def _build_guardrail_lines(trace: ExecutionTrace) -> List[str]:
    return [
        _example_line(trace),
        _replan_line(trace),
        _unknown_intent_line(trace.page_intents),
    ]


def _lifecycle_lines(trace: ExecutionTrace) -> List[str]:
    lines: List[str] = []
    for event in trace.learning_events:
        if not getattr(event, "event", None):
            continue
        label = str(event.event)
        detail = event.data or {}
        reason = detail.get("reason") if isinstance(detail, dict) else None
        subgoal = detail.get("subgoal_id") if isinstance(detail, dict) else None
        suffix_bits = []
        if reason:
            suffix_bits.append(f"reason={reason}")
        if subgoal:
            suffix_bits.append(f"subgoal={subgoal}")
        suffix = f" ({', '.join(suffix_bits)})" if suffix_bits else ""
        if label == "mission_halted":
            lines.append(f"Agent halted execution due to autonomous judgment or approval gating{suffix}.")
        elif label == "resume_loaded":
            checkpoint = detail.get("checkpoint") if isinstance(detail, dict) else None
            checkpoint_msg = f" from {checkpoint}" if checkpoint else ""
            lines.append(f"Resume loaded{checkpoint_msg}.")
        elif label == "resume_completed":
            lines.append("Mission resumed and completed successfully.")
        elif label == "escalation_requested":
            lines.append("🔶 Escalation requested after risk budget breach.")
        elif label == "escalation_granted":
            lines.append("🔓 Escalation granted with bounded expanded budget.")
        elif label == "escalation_closed":
            lines.append("🔒 Escalation window closed; execution returned to base budget.")
        elif label == "escalation_exploration":
            lines.append("🧭 Exploration continued under escalation window.")
        elif label == "demo_force_actions":
            lines.append("Demo preflight executed (forced navigation/search/scroll).")
    return lines


def _capability_report_lines(trace: ExecutionTrace) -> List[str]:
    report = trace.capability_report or {}
    if not report:
        return []
    risk_level = str(report.get("risk_level") or "unknown").lower()
    missing_entries = report.get("missing") or []
    missing_ids = _joined(
        str(entry.get("capability_id"))
        for entry in missing_entries
        if isinstance(entry, dict)
    )
    lines = [f"Plan capability risk: {risk_level}"]
    lines.append("Missing capabilities: none" if not missing_ids else f"Missing capabilities: {missing_ids}")
    return lines


def _capabilities_lines(trace: ExecutionTrace) -> List[str]:
    usages = trace.capabilities_used
    if not usages:
        return ["Capabilities used: none"]
    seen = set()
    lines = ["Capabilities used:"]
    for usage in usages:
        key = (usage.capability_id, usage.skill_id, usage.subgoal_id)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {usage.capability_id} (via {usage.skill_id})")
    return lines


def _capability_enforcement_lines(trace: ExecutionTrace) -> List[str]:
    decisions = trace.capability_enforcements
    if not decisions:
        return ["Capability enforcement: none"]
    lines = ["Capability enforcement:"]
    seen = set()
    for decision in decisions:
        key = (decision.capability_id, decision.decision, decision.subgoal_id)
        if key in seen:
            continue
        seen.add(key)
        detail = f"{decision.capability_id} -> {decision.decision.upper()} (conf {decision.confidence:.2f} < {decision.threshold:.2f}?)"
        if decision.reason:
            detail += f" reason={decision.reason}"
        if decision.subgoal_id:
            detail += f" subgoal={decision.subgoal_id}"
        lines.append(f"- {detail}")
    return lines


def _approval_lines(trace: ExecutionTrace) -> List[str]:
    if not trace.approvals_requested and not trace.approvals_resolved:
        return []
    lines: List[str] = []
    for request in trace.approvals_requested:
        detail = f"⚠️ Human approval requested: {request.requested_action.get('name', 'action')} ({request.risk_level})"
        lines.append(detail)
    for resolution in trace.approvals_resolved:
        symbol = "✅" if resolution.state == "approved" else "⛔"
        detail = f"{symbol} Approval {resolution.state} for {resolution.subgoal_id}"
        lines.append(detail)
    return lines


def _learning_explanation_line(trace: ExecutionTrace) -> Optional[str]:
    mission_dir: Optional[Path] = None
    for artifact in trace.artifacts:
        if artifact.name == "mission_dir":
            mission_dir = Path(artifact.path)
            break
    for artifact in trace.artifacts:
        if artifact.name == "learning_decision_explanation":
            path_value = Path(artifact.path)
            if mission_dir:
                try:
                    path_value = path_value.relative_to(mission_dir)
                except ValueError:
                    path_value = Path(artifact.path)
            return f"Learning decision explanation: {path_value.as_posix()}"
    return None


def _example_line(trace: ExecutionTrace) -> str:
    mentions_example = any(
        "example.com" in (text or "").lower()
        for text in _iter_trace_text(trace)
    )
    return "Example.com references: none" if not mentions_example else "Example.com references detected"


def _replan_line(trace: ExecutionTrace) -> str:
    attempt_replans = sum(1 for subgoal in trace.subgoal_traces if (subgoal.attempt_number or 0) > 1)
    eval_replans = int(trace.evaluation.get("replans", 0)) if isinstance(trace.evaluation, dict) else 0
    replans = max(attempt_replans, eval_replans)
    return "Planner replans triggered: none" if replans == 0 else f"Planner replans triggered: {replans}"


def _unknown_intent_line(page_intents: List[PageIntentRecord]) -> str:
    unknown = any((intent.intent or "").strip().upper() == "UNKNOWN" for intent in page_intents)
    return "Unknown intents observed: none" if not unknown else "Unknown intents observed"


def _extract_url(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"https?://[^\s)]+", text)
    return match.group(0) if match else None


def _iter_trace_text(trace: ExecutionTrace) -> Iterable[str]:
    yield trace.mission_text
    for subgoal in trace.subgoal_traces:
        yield subgoal.description
        if subgoal.error:
            yield subgoal.error
    for artifact in trace.artifacts:
        yield artifact.path
    for failure in trace.failures:
        yield failure.message
    for warning in trace.warnings:
        yield warning
