"""Data models for mission planning and execution."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

UTC = timezone.utc

MissionStatus = Literal[
    "pending",
    "running",
    "complete",
    "failed",
    "skipped",
    "halted",
    "ask_human",
    "refused_by_learning",
]
SubgoalStatus = MissionStatus


def mission_id(prefix: str = "mission") -> str:
    """Return a timestamp-derived mission identifier."""

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{timestamp}"


class MissionSpec(BaseModel):
    """User-provided mission definition."""

    id: str = Field(default_factory=mission_id)
    instruction: str
    constraints: Optional[Dict[str, Any]] = None
    timeout_secs: int = Field(default=900, ge=60, le=21600)
    max_retries: int = Field(default=2, ge=0, le=5)
    allow_sensitive: bool = False
    execute: bool = False
    autonomy_budget: Optional[Dict[str, Any]] = None
    safety_contract: Optional[Dict[str, Any]] = None
    ask_on_uncertainty: bool = False
    learning_review: bool = False

    @field_validator("instruction")
    @classmethod
    def _validate_instruction(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("instruction cannot be empty")
        return value.strip()


class MissionSubgoal(BaseModel):
    """One planner-derived mission subgoal."""

    id: str
    description: str
    planner_metadata: Dict[str, Any] = Field(default_factory=dict)


class MissionSubgoalResult(BaseModel):
    """Execution payload captured for each completed subgoal."""

    subgoal_id: str
    description: str
    status: SubgoalStatus
    attempts: int
    started_at: datetime
    ended_at: datetime
    completion: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)


class MissionResult(BaseModel):
    """Top-level mission outcome payload."""

    mission_id: str
    status: MissionStatus
    start_ts: datetime
    end_ts: datetime
    subgoal_results: List[MissionSubgoalResult]
    summary: Dict[str, Any] = Field(default_factory=dict)
    artifacts_path: str
    termination: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "MissionSpec",
    "MissionSubgoal",
    "MissionSubgoalResult",
    "MissionResult",
    "MissionStatus",
    "mission_id",
]
