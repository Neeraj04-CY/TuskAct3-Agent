from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .escalation_state import EscalationState


@dataclass
class ResumeCheckpoint:
    mission_id: str
    halted_subgoal_id: str
    halted_reason: str
    page_url: Optional[str]
    page_intent: Optional[str]
    completed_subgoals: List[str]
    pending_subgoals: List[str]
    skills_used: List[str]
    capability_state: Dict[str, Any]
    learning_bias_snapshot: Dict[str, Any]
    trace_path: str
    timestamp_utc: str
    escalation_state: Dict[str, Any] = field(default_factory=lambda: EscalationState().to_dict())

    # Optional metadata to help resume in process restarts.
    mission_instruction: Optional[str] = None
    artifacts_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, source: Path) -> "ResumeCheckpoint":
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        if payload.get("escalation_state") is None:
            payload["escalation_state"] = EscalationState().to_dict()
        return cls(**payload)


__all__ = ["ResumeCheckpoint"]
