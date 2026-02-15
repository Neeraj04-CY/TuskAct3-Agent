from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from eikon_engine.capabilities.models import CapabilityId

from eikon_engine.missions.mission_schema import MissionSpec, MissionStatus, MissionSubgoal

from .models import (
    ActionTrace,
    ArtifactRecord,
    ExecutionTrace,
    FailureRecord,
    PageIntentRecord,
    ExtractionRecord,
    SkillUsage,
    SubgoalSkipRecord,
    SubgoalTrace,
    TRACE_VERSION,
    LearningEventRecord,
    CapabilityUsage,
    CapabilityEnforcementDecision,
    ApprovalRequestRecord,
    ApprovalResolutionRecord,
)
from .serializer import ExecutionTraceSerializer

UTC = timezone.utc


def _now() -> datetime:
    return datetime.now(UTC)


def _build_trace_id(mission_id: str, started_at: datetime) -> str:
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in mission_id)
    return f"trace_{timestamp}_{sanitized}"


def _duration_ms(started_at: datetime, ended_at: datetime | None) -> Optional[float]:
    if not ended_at:
        return None
    return max((ended_at - started_at).total_seconds() * 1000, 0.0)


class ExecutionTraceRecorder:
    """Stateful recorder that accumulates mission execution traces."""

    def __init__(
        self,
        *,
        storage_dir: Path | str | None = None,
        serializer: ExecutionTraceSerializer | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir or "traces")
        self.serializer = serializer or ExecutionTraceSerializer()
        self._trace: ExecutionTrace | None = None
        self._trace_path: Path | None = None
        self._subgoal_handles: Dict[str, SubgoalTrace] = {}
        self._last_action_end: Dict[str, datetime] = {}

    @property
    def trace_path(self) -> Path | None:
        return self._trace_path

    @property
    def trace(self) -> ExecutionTrace | None:
        return self._trace

    def start(self, *, mission_spec: MissionSpec, mission_dir: Path, started_at: datetime) -> None:
        if self._trace is not None:
            raise RuntimeError("execution trace already started")
        trace_id = _build_trace_id(mission_spec.id, started_at)
        self._trace = ExecutionTrace(
            id=trace_id,
            mission_id=mission_spec.id,
            mission_text=mission_spec.instruction,
            type="execution_trace",
            started_at=started_at,
            status="running",
            trace_version=TRACE_VERSION,
            evaluation={"phase": "pending"},
        )
        self.record_artifact("mission_dir", str(mission_dir))

    def start_subgoal(
        self,
        *,
        subgoal: MissionSubgoal,
        attempt_number: int,
        learning_bias: Optional[Dict[str, Any]] = None,
        learning_score: float | None = None,
        learning_bias_snapshot: Optional[Dict[str, Any]] = None,
        impact_score_at_decision: float | None = None,
        override_reason: str | None = None,
        refusal_reason: str | None = None,
        capability_requirements: Optional[List[Dict[str, Any]]] = None,
        capability_enforcements: Optional[List[CapabilityEnforcementDecision]] = None,
    ) -> str:
        trace = self._require_trace()
        handle = f"{subgoal.id}__attempt_{attempt_number}"
        subgoal_trace = SubgoalTrace(
            id=handle,
            subgoal_id=subgoal.id,
            description=subgoal.description,
            attempt_number=attempt_number,
            started_at=_now(),
            type="subgoal_trace",
            status="running",
            learning_bias=dict(learning_bias or {}),
            learning_score=learning_score,
            learning_bias_snapshot=dict(learning_bias_snapshot or {}),
            impact_score_at_decision=impact_score_at_decision,
            override_reason=override_reason,
            refusal_reason=refusal_reason,
            capability_requirements=list(capability_requirements or []),
            capability_enforcements=list(capability_enforcements or []),
        )
        trace.subgoal_traces.append(subgoal_trace)
        self._subgoal_handles[handle] = subgoal_trace
        self._last_action_end[handle] = subgoal_trace.started_at
        return handle

    def end_subgoal(
        self,
        handle: str,
        *,
        status: MissionStatus,
        error: str | None = None,
        ended_at: datetime | None = None,
    ) -> None:
        subgoal_trace = self._subgoal_handles.get(handle)
        if not subgoal_trace:
            return
        subgoal_trace.status = status
        subgoal_trace.error = error
        subgoal_trace.ended_at = ended_at or _now()
        subgoal_trace.duration_ms = _duration_ms(subgoal_trace.started_at, subgoal_trace.ended_at)

    def record_action(
        self,
        handle: str | None,
        *,
        action_type: str,
        selector: str | None,
        target: str | None,
        input_data: str | None,
        status: str,
        started_at: datetime,
        ended_at: datetime | None,
        duration_ms: float | None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not handle:
            return
        subgoal_trace = self._subgoal_handles.get(handle)
        if not subgoal_trace:
            return
        sequence = len(subgoal_trace.actions_taken) + 1
        action_id = f"{subgoal_trace.id}_action_{sequence:03d}"
        action_trace = ActionTrace(
            id=action_id,
            sequence=sequence,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            action_type=action_type,
            selector=selector,
            target=target,
            input_data=input_data,
            status=status,
            metadata=dict(metadata or {}),
        )
        subgoal_trace.actions_taken.append(action_trace)
        if ended_at:
            self._last_action_end[handle] = ended_at

    def record_skill_usage(
        self,
        *,
        name: str,
        status: str,
        handle: str | None,
        metadata: Optional[Dict[str, Any]] = None,
        learning_bias: Optional[Dict[str, Any]] = None,
        prior_success_rate: float | None = None,
        learning_weight_applied: bool = False,
        preferred_via_learning: bool = False,
    ) -> None:
        trace = self._require_trace()
        attempt_number: Optional[int] = None
        subgoal_id: Optional[str] = None
        if handle and handle in self._subgoal_handles:
            subgoal = self._subgoal_handles[handle]
            subgoal.skill_used = name
            subgoal_id = subgoal.subgoal_id
            attempt_number = subgoal.attempt_number
        event_time = _now()
        entry_id = f"skill_{len(trace.skills_used) + 1:03d}"
        bias_snapshot = dict(learning_bias or {})
        usage = SkillUsage(
            id=entry_id,
            name=name,
            status=status,
            subgoal_id=subgoal_id,
            attempt_number=attempt_number,
            started_at=event_time,
            ended_at=event_time,
            duration_ms=0.0,
            metadata=dict(metadata or {}),
            learning_bias=bias_snapshot or None,
            learning_bias_applied=bool(bias_snapshot),
            bias_snapshot=bias_snapshot or None,
            prior_success_rate=prior_success_rate,
            learning_weight_applied=learning_weight_applied,
            preferred_via_learning=preferred_via_learning,
        )
        trace.skills_used.append(usage)

    def record_capability_usage(
        self,
        *,
        skill_id: str,
        handle: str | None,
        capability_ids: List[CapabilityId],
        confidence: float | None = None,
    ) -> None:
        trace = self._require_trace()
        if not handle or handle not in self._subgoal_handles:
            return
        subgoal = self._subgoal_handles[handle]
        for capability_id in capability_ids:
            usage = CapabilityUsage(
                capability_id=capability_id,
                skill_id=skill_id,
                subgoal_id=subgoal.subgoal_id,
                confidence=confidence,
            )
            subgoal.capabilities_used.append(usage)
            trace.capabilities_used.append(usage)

    def record_capability_enforcements(
        self,
        *,
        handle: str | None,
        decisions: List[CapabilityEnforcementDecision],
    ) -> None:
        trace = self._require_trace()
        if handle and handle in self._subgoal_handles:
            subgoal = self._subgoal_handles[handle]
            subgoal.capability_enforcements.extend(decisions)
        trace.capability_enforcements.extend(decisions)

    def record_learning_event(self, *, event: str, data: Dict[str, Any]) -> None:
        trace = self._require_trace()
        entry_id = f"learning_{len(trace.learning_events) + 1:03d}"
        trace.learning_events.append(
            LearningEventRecord(
                id=entry_id,
                event=event,
                data=dict(data or {}),
            )
        )

    def record_lifecycle_event(self, *, event: str, data: Dict[str, Any]) -> None:
        trace = self._require_trace()
        entry_id = f"lifecycle_{len(trace.learning_events) + 1:03d}"
        trace.learning_events.append(
            LearningEventRecord(
                id=entry_id,
                event=event,
                data=dict(data or {}),
            )
        )

    def record_warning(self, message: str) -> None:
        trace = self._require_trace()
        trace.warnings.append(message)

    def record_failure(
        self,
        *,
        failure_type: str,
        message: str,
        handle: str | None = None,
        retryable: bool = False,
    ) -> None:
        trace = self._require_trace()
        subgoal_id: Optional[str] = None
        timestamp = _now()
        if handle and handle in self._subgoal_handles:
            subgoal = self._subgoal_handles[handle]
            subgoal_id = subgoal.subgoal_id
            last_action_time = self._last_action_end.get(handle)
            if last_action_time:
                timestamp = max(timestamp, last_action_time)
        failure_id = f"failure_{len(trace.failures) + 1:03d}"
        trace.failures.append(
            FailureRecord(
                id=failure_id,
                failure_type=failure_type,
                message=message,
                subgoal_id=subgoal_id,
                retryable=retryable,
                started_at=timestamp,
                ended_at=timestamp,
                duration_ms=0.0,
            )
        )

    def record_artifact(self, name: str, path_value: str) -> None:
        trace = self._require_trace()
        timestamp = _now()
        entry_id = f"artifact_{len(trace.artifacts) + 1:03d}"
        trace.artifacts.append(
            ArtifactRecord(
                id=entry_id,
                name=name,
                path=path_value,
                started_at=timestamp,
                ended_at=timestamp,
                duration_ms=0.0,
            )
        )

    def record_capability_report(self, capability_report: Dict[str, Any] | None) -> None:
        trace = self._require_trace()
        if capability_report is None:
            return
        trace.capability_report = dict(capability_report)

    def record_page_intent(
        self,
        *,
        intent: str,
        confidence: float,
        strategy: str | None = None,
        signals: Optional[Dict[str, Any]] = None,
        step_id: str | None = None,
    ) -> None:
        trace = self._require_trace()
        entry_id = f"intent_{len(trace.page_intents) + 1:03d}"
        trace.page_intents.append(
            PageIntentRecord(
                id=entry_id,
                intent=intent,
                strategy=strategy,
                confidence=confidence,
                signals=dict(signals or {}),
                step_id=step_id,
            )
        )

    def record_extraction(
        self,
        *,
        name: str,
        status: str,
        summary: Dict[str, Any],
        artifact_path: str | None = None,
    ) -> None:
        trace = self._require_trace()
        entry_id = f"extraction_{len(trace.extractions) + 1:03d}"
        timestamp = _now()
        trace.extractions.append(
            ExtractionRecord(
                id=entry_id,
                name=name,
                status=status,
                summary=dict(summary),
                artifact_path=artifact_path,
                started_at=timestamp,
                ended_at=timestamp,
                duration_ms=0.0,
            )
        )

    def record_subgoal_skip(
        self,
        *,
        subgoal: MissionSubgoal,
        reason: str,
        page_intent: str | None = None,
    ) -> None:
        trace = self._require_trace()
        entry_id = f"skip_{len(trace.skipped_subgoals) + 1:03d}"
        trace.skipped_subgoals.append(
            SubgoalSkipRecord(
                id=entry_id,
                subgoal_id=subgoal.id,
                description=subgoal.description,
                reason=reason,
                page_intent=page_intent,
            )
        )

    def record_approval_request(
        self,
        *,
        approval_id: str,
        subgoal_id: str,
        reason: str,
        risk_level: str,
        requested_action: Dict[str, Any],
        expires_at: datetime | None,
    ) -> None:
        trace = self._require_trace()
        entry_id = f"approval_req_{len(trace.approvals_requested) + 1:03d}"
        trace.approvals_requested.append(
            ApprovalRequestRecord(
                id=entry_id,
                approval_id=approval_id,
                subgoal_id=subgoal_id,
                reason=reason,
                risk_level=risk_level,
                requested_action=requested_action,
                expires_at=expires_at,
            )
        )

    def record_approval_resolution(
        self,
        *,
        approval_id: str,
        subgoal_id: str,
        state: str,
        resolved_by: str | None,
        reason: str | None,
        external: bool = False,
    ) -> None:
        trace = self._require_trace()
        entry_id = f"approval_res_{len(trace.approvals_resolved) + 1:03d}"
        trace.approvals_resolved.append(
            ApprovalResolutionRecord(
                id=entry_id,
                approval_id=approval_id,
                subgoal_id=subgoal_id,
                state=state,  # type: ignore[arg-type]
                resolved_by=resolved_by,
                reason=reason,
                external=external,
            )
        )

    def finalize(self, *, status: MissionStatus, ended_at: datetime | None = None) -> None:
        trace = self._require_trace()
        trace.status = status
        trace.ended_at = ended_at or _now()
        trace.duration_ms = _duration_ms(trace.started_at, trace.ended_at)
        self._sort_trace_entries()
        warnings = self._validate_trace(trace)
        if warnings:
            trace.incomplete = True
            trace.warnings.extend(warnings)

    def persist(self) -> Path:
        trace = self._require_trace()
        self._sort_trace_entries()
        path = self.serializer.save(trace, directory=self.storage_dir)
        self._trace_path = path
        return path

    def _require_trace(self) -> ExecutionTrace:
        if not self._trace:
            raise RuntimeError("execution trace not initialized")
        return self._trace

    def _sort_trace_entries(self) -> None:
        trace = self._trace
        if not trace:
            return
        trace.subgoal_traces.sort(key=lambda item: item.started_at)
        for subgoal in trace.subgoal_traces:
            subgoal.actions_taken.sort(key=lambda action: action.sequence)
        trace.skills_used.sort(key=lambda usage: usage.started_at)
        trace.failures.sort(key=lambda failure: failure.started_at)
        trace.artifacts.sort(key=lambda artifact: artifact.started_at)
        trace.page_intents.sort(key=lambda record: record.decided_at)
        trace.extractions.sort(key=lambda record: record.started_at)
        trace.skipped_subgoals.sort(key=lambda record: record.decided_at)
        trace.learning_events.sort(key=lambda record: record.occurred_at)
        trace.approvals_requested.sort(key=lambda record: record.created_at)
        trace.approvals_resolved.sort(key=lambda record: record.resolved_at)
        for subgoal in trace.subgoal_traces:
            subgoal.capabilities_used.sort(key=lambda entry: (entry.capability_id, entry.skill_id))
        trace.capabilities_used.sort(key=lambda entry: (entry.capability_id, entry.skill_id))
        trace.capability_enforcements.sort(
            key=lambda entry: (
                entry.capability_id,
                entry.subgoal_id or "",
                entry.decision,
                entry.confidence,
            )
        )

    def _validate_trace(self, trace: ExecutionTrace) -> List[str]:
        warnings: List[str] = []
        for subgoal in trace.subgoal_traces:
            if subgoal.ended_at is None:
                warnings.append(f"Subgoal {subgoal.id} missing end timestamp")
            if subgoal.status != "complete" and not subgoal.error:
                warnings.append(f"Subgoal {subgoal.id} missing failure reason")
            for action in subgoal.actions_taken:
                if not action.status:
                    warnings.append(f"Action {action.id} missing status")
        for failure in trace.failures:
            if failure.retryable and not failure.message:
                warnings.append(f"Failure {failure.id} missing retry message")
        for usage in trace.skills_used:
            if usage.subgoal_id is None:
                warnings.append(f"Skill usage {usage.id} missing subgoal reference")
        return warnings


__all__ = ["ExecutionTraceRecorder"]
