from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


@dataclass
class LearningSkillUsage:
    skill_name: str
    success: bool
    steps_saved: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "success": self.success,
            "steps_saved": int(self.steps_saved),
        }


@dataclass
class LearningFailure:
    step: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "reason": self.reason,
        }


@dataclass
class LearningRecord:
    mission_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    mission_type: str = "unknown"
    skills_used: List[LearningSkillUsage] = field(default_factory=list)
    failures: List[LearningFailure] = field(default_factory=list)
    confidence_score: float = 0.0
    outcome: str = "unknown"
    resumed: bool = False
    resume_source: str | None = None
    escalation_used: bool = False
    escalation_outcome: str | None = None
    trace_id: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "timestamp": self.timestamp,
            "mission_type": self.mission_type,
            "skills_used": [entry.to_dict() for entry in self.skills_used],
            "failures": [entry.to_dict() for entry in self.failures],
            "confidence_score": float(self.confidence_score),
            "outcome": self.outcome,
            "resumed": self.resumed,
            "resume_source": self.resume_source,
            "escalation_used": self.escalation_used,
            "escalation_outcome": self.escalation_outcome,
            "trace_id": self.trace_id,
        }


__all__ = ["LearningRecord", "LearningSkillUsage", "LearningFailure"]
