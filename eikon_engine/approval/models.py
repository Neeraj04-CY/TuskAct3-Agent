from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Literal, Optional

ApprovalState = Literal["not_required", "pending", "approved", "rejected", "expired"]
UTC = timezone.utc


@dataclass
class ApprovalRequest:
    approval_id: str
    mission_id: str
    subgoal_id: str
    requested_action: Dict[str, Any]
    reason: str
    risk_level: str
    capabilities_required: List[str]
    learning_bias: Dict[str, Any]
    alternatives: List[str]
    expires_at: datetime
    state: ApprovalState = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_reason: Optional[str] = None
    approved_by_human: bool = False
    external: bool = False

    def to_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["expires_at"] = self.expires_at.isoformat()
        payload["created_at"] = self.created_at.isoformat()
        payload["resolved_at"] = self.resolved_at.isoformat() if self.resolved_at else None
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ApprovalRequest":
        expires_at = cls._parse_ts(payload.get("expires_at"))
        created_at = cls._parse_ts(payload.get("created_at")) or datetime.now(UTC)
        resolved_at = cls._parse_ts(payload.get("resolved_at"))
        return cls(
            approval_id=str(payload.get("approval_id")),
            mission_id=str(payload.get("mission_id")),
            subgoal_id=str(payload.get("subgoal_id")),
            requested_action=dict(payload.get("requested_action") or {}),
            reason=str(payload.get("reason") or "approval_required"),
            risk_level=str(payload.get("risk_level") or "medium"),
            capabilities_required=list(payload.get("capabilities_required") or []),
            learning_bias=dict(payload.get("learning_bias") or {}),
            alternatives=list(payload.get("alternatives") or []),
            expires_at=expires_at or datetime.now(UTC) + timedelta(minutes=10),
            state=str(payload.get("state") or "pending"),
            created_at=created_at,
            resolved_at=resolved_at,
            resolved_by=payload.get("resolved_by"),
            resolution_reason=payload.get("resolution_reason"),
            approved_by_human=bool(payload.get("approved_by_human", False)),
            external=bool(payload.get("external", False)),
        )

    @staticmethod
    def _parse_ts(value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None
