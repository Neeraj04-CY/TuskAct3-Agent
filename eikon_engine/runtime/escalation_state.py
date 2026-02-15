from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

UTC = timezone.utc


@dataclass
class EscalationState:
    allowed: bool = True
    used: bool = False
    reason: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    expanded_budget: Dict[str, Any] = field(default_factory=dict)
    window_limits: Dict[str, Any] = field(default_factory=dict)
    actions_tagged: bool = False

    def mark_requested(self, reason: str, *, expanded_budget: Dict[str, Any], window_limits: Dict[str, Any]) -> None:
        now = datetime.now(UTC).isoformat()
        self.used = True
        self.reason = reason
        self.started_at = now
        self.expanded_budget = dict(expanded_budget)
        self.window_limits = dict(window_limits)

    def mark_closed(self, reason: str | None = None) -> None:
        self.ended_at = datetime.now(UTC).isoformat()
        if reason and not self.reason:
            self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "EscalationState":
        if not payload:
            return cls()
        return cls(**payload)


__all__ = ["EscalationState"]
