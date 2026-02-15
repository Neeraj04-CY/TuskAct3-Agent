"""Execution trace data models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from eikon_engine.capabilities.models import CapabilityId
from eikon_engine.approval.models import ApprovalState

from pydantic import BaseModel, Field

TraceStatus = Literal["running", "complete", "failed", "aborted", "refused_by_learning", "halted"]
TRACE_VERSION = "v3.1"
UTC = timezone.utc


def _now() -> datetime:
    return datetime.now(UTC)


def _duration_ms(started_at: datetime, ended_at: datetime | None) -> Optional[float]:
    if not ended_at:
        return None
    return max((ended_at - started_at).total_seconds() * 1000, 0.0)


class ActionTrace(BaseModel):
    """Atomic browser/tool action recorded when BrowserWorker runs a step."""

    id: str
    type: Literal["action_trace"] = "action_trace"
    sequence: int
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    action_type: str
    selector: Optional[str] = None
    target: Optional[str] = None
    input_data: Optional[str] = None
    status: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FailureRecord(BaseModel):
    """Failure event captured immediately after the triggering action."""

    id: str
    type: Literal["failure_record"] = "failure_record"
    started_at: datetime = Field(default_factory=_now)
    ended_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    failure_type: str
    message: str
    subgoal_id: Optional[str] = None
    retryable: bool = False


class SkillUsage(BaseModel):
    """Skill invocation snapshot taken when a skill begins."""

    id: str
    type: Literal["skill_usage"] = "skill_usage"
    started_at: datetime = Field(default_factory=_now)
    ended_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    name: str
    status: str
    subgoal_id: Optional[str] = None
    attempt_number: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    learning_bias: Optional[Dict[str, Any]] = None
    learning_bias_applied: bool = False
    bias_snapshot: Optional[Dict[str, Any]] = None
    prior_success_rate: Optional[float] = None
    learning_weight_applied: bool = False
    preferred_via_learning: bool = False


class LearningEventRecord(BaseModel):
    """Learning override and conflict events applied to the mission."""

    id: str
    type: Literal["learning_event"] = "learning_event"
    event: str
    data: Dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=_now)


class ArtifactRecord(BaseModel):
    """Pointer to a file emitted during execution."""

    id: str
    type: Literal["artifact_record"] = "artifact_record"
    started_at: datetime = Field(default_factory=_now)
    ended_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    name: str
    path: str


class PageIntentRecord(BaseModel):
    """Decision record that captures the detected page intent and selected strategy."""

    id: str
    type: Literal["page_intent_record"] = "page_intent_record"
    intent: str
    strategy: Optional[str] = None
    confidence: float
    signals: Dict[str, Any] = Field(default_factory=dict)
    step_id: Optional[str] = None
    decided_at: datetime = Field(default_factory=_now)


class ExtractionRecord(BaseModel):
    """Structured extraction summary emitted by higher-level skills."""

    id: str
    type: Literal["extraction_record"] = "extraction_record"
    name: str
    status: str
    summary: Dict[str, Any] = Field(default_factory=dict)
    artifact_path: Optional[str] = None
    started_at: datetime = Field(default_factory=_now)
    ended_at: Optional[datetime] = None
    duration_ms: Optional[float] = None


class SubgoalSkipRecord(BaseModel):
    """Log entry recorded when a subgoal is skipped due to intent gating."""

    id: str
    type: Literal["subgoal_skip"] = "subgoal_skip"
    subgoal_id: str
    description: str
    reason: str
    page_intent: Optional[str] = None
    decided_at: datetime = Field(default_factory=_now)


class ApprovalRequestRecord(BaseModel):
    """Approval request emitted when execution pauses for human review."""

    id: str
    type: Literal["approval_request"] = "approval_request"
    approval_id: str
    subgoal_id: str
    reason: str
    risk_level: str
    requested_action: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)
    expires_at: Optional[datetime] = None


class ApprovalResolutionRecord(BaseModel):
    """Approval resolution event recorded after human decision or timeout."""

    id: str
    type: Literal["approval_resolution"] = "approval_resolution"
    approval_id: str
    subgoal_id: str
    state: ApprovalState
    resolved_at: datetime = Field(default_factory=_now)
    resolved_by: Optional[str] = None
    reason: Optional[str] = None
    external: bool = False


@dataclass
class CapabilityUsage:
    capability_id: CapabilityId
    skill_id: str
    subgoal_id: str
    confidence: Optional[float] = None


@dataclass
class CapabilityEnforcementDecision:
    capability_id: CapabilityId
    decision: Literal["allow", "warn_only", "ask_human"]
    confidence: float
    threshold: float
    critical: float
    reason: str
    required: bool
    missing: bool
    subgoal_id: Optional[str] = None
    source: Optional[str] = None


class SubgoalTrace(BaseModel):
    """One attempt of a mission subgoal."""

    id: str
    type: Literal["subgoal_trace"] = "subgoal_trace"
    subgoal_id: str
    description: str
    attempt_number: int
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: Optional[TraceStatus] = None
    actions_taken: List[ActionTrace] = Field(default_factory=list)
    skill_used: Optional[str] = None
    error: Optional[str] = None
    learning_bias: Optional[Dict[str, Any]] = None
    learning_score: Optional[float] = None
    learning_bias_snapshot: Optional[Dict[str, Any]] = None
    impact_score_at_decision: Optional[float] = None
    override_reason: Optional[str] = None
    refusal_reason: Optional[str] = None
    capabilities_used: List[CapabilityUsage] = Field(default_factory=list)
    capability_requirements: List[Dict[str, Any]] = Field(default_factory=list)
    capability_enforcements: List[CapabilityEnforcementDecision] = Field(default_factory=list)


class ExecutionTrace(BaseModel):
    """Top-level mission run trace."""

    trace_version: str = TRACE_VERSION
    id: str
    type: Literal["execution_trace"] = "execution_trace"
    mission_id: str
    mission_text: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: TraceStatus = "running"
    subgoal_traces: List[SubgoalTrace] = Field(default_factory=list)
    skills_used: List[SkillUsage] = Field(default_factory=list)
    failures: List[FailureRecord] = Field(default_factory=list)
    artifacts: List[ArtifactRecord] = Field(default_factory=list)
    page_intents: List[PageIntentRecord] = Field(default_factory=list)
    extractions: List[ExtractionRecord] = Field(default_factory=list)
    skipped_subgoals: List[SubgoalSkipRecord] = Field(default_factory=list)
    approvals_requested: List[ApprovalRequestRecord] = Field(default_factory=list)
    approvals_resolved: List[ApprovalResolutionRecord] = Field(default_factory=list)
    learning_events: List[LearningEventRecord] = Field(default_factory=list)
    capabilities_used: List[CapabilityUsage] = Field(default_factory=list)
    capability_report: Dict[str, Any] = Field(default_factory=dict)
    capability_enforcements: List[CapabilityEnforcementDecision] = Field(default_factory=list)
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    incomplete: bool = False
    warnings: List[str] = Field(default_factory=list)


__all__ = [
    "TRACE_VERSION",
    "TraceStatus",
    "ActionTrace",
    "FailureRecord",
    "SkillUsage",
    "ArtifactRecord",
    "PageIntentRecord",
    "ExtractionRecord",
    "SubgoalSkipRecord",
    "LearningEventRecord",
    "SubgoalTrace",
    "ExecutionTrace",
    "CapabilityUsage",
    "ApprovalRequestRecord",
    "ApprovalResolutionRecord",
]
