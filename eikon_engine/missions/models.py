from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional

UTC = timezone.utc


class MissionTermination(Enum):
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    HALTED = "HALTED"
    ASK_HUMAN = "ASK_HUMAN"


@dataclass
class AutonomyBudget:
    max_steps: int
    max_retries: int
    max_duration_sec: float
    max_risk_score: float


DEFAULT_AUTONOMY_BUDGET = AutonomyBudget(
    max_steps=30,
    max_retries=3,
    max_duration_sec=120.0,
    max_risk_score=0.4,
)


@dataclass
class SafetyContract:
    allowed_actions: Optional[list[str]] = None
    blocked_actions: Optional[list[str]] = None
    requires_confirmation: bool = False

    def normalize(self) -> "SafetyContract":
        allowed = [action.lower() for action in (self.allowed_actions or [])]
        blocked = [action.lower() for action in (self.blocked_actions or [])]
        return SafetyContract(
            allowed_actions=allowed or None,
            blocked_actions=blocked or None,
            requires_confirmation=self.requires_confirmation,
        )


@dataclass
class BudgetUsage:
    steps_used: int = 0
    retries_used: int = 0
    risk_score: float = 0.0
    confidence_samples: int = 0
    confidence_total: float = 0.0
    failure_events: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def elapsed_seconds(self) -> float:
        return max(0.0, (datetime.now(UTC) - self.started_at).total_seconds())

    def average_confidence(self) -> Optional[float]:
        if not self.confidence_samples:
            return None
        return round(self.confidence_total / self.confidence_samples, 3)

    def snapshot(self) -> Dict[str, float | int]:
        return {
            "steps_used": self.steps_used,
            "retries_used": self.retries_used,
            "risk_score": round(self.risk_score, 3),
            "confidence_samples": self.confidence_samples,
            "failure_events": self.failure_events,
            "elapsed_seconds": round(self.elapsed_seconds(), 2),
        }


class BudgetMonitor:
    def __init__(self, budget: AutonomyBudget) -> None:
        self.budget = budget
        self.usage = BudgetUsage()

    def record_steps(self, steps: int) -> None:
        if steps <= 0:
            return
        self.usage.steps_used += steps

    def record_retry(self) -> None:
        self.usage.retries_used += 1
        self._recalculate_risk()

    def record_failure(self) -> None:
        self.usage.failure_events += 1
        self._recalculate_risk()

    def record_confidence(self, confidence: float | None) -> None:
        if confidence is None:
            return
        clamped = max(0.0, min(1.0, float(confidence)))
        self.usage.confidence_samples += 1
        self.usage.confidence_total += clamped
        self._recalculate_risk()

    def _recalculate_risk(self) -> None:
        avg_conf = self.usage.average_confidence()
        confidence_penalty = 0.0
        if avg_conf is not None:
            confidence_penalty = max(0.0, 0.7 - avg_conf)
        failure_penalty = min(0.5, 0.15 * self.usage.failure_events)
        retry_penalty = min(0.3, 0.05 * self.usage.retries_used)
        self.usage.risk_score = round(confidence_penalty + failure_penalty + retry_penalty, 3)

    def limits_exceeded(self) -> tuple[bool, Optional[str], Dict[str, float | int]]:
        if self.usage.steps_used > self.budget.max_steps:
            return True, "max_steps_exceeded", {
                "steps_used": self.usage.steps_used,
                "limit": self.budget.max_steps,
            }
        if self.usage.retries_used > self.budget.max_retries:
            return True, "max_retries_exceeded", {
                "retries_used": self.usage.retries_used,
                "limit": self.budget.max_retries,
            }
        if self.usage.elapsed_seconds() > self.budget.max_duration_sec:
            return True, "duration_exceeded", {
                "elapsed_seconds": round(self.usage.elapsed_seconds(), 2),
                "limit": self.budget.max_duration_sec,
            }
        if self.usage.risk_score > self.budget.max_risk_score:
            return True, "risk_budget_exceeded", {
                "risk_score": self.usage.risk_score,
                "limit": self.budget.max_risk_score,
            }
        return False, None, {}

    def snapshot(self) -> Dict[str, float | int]:
        snapshot = self.usage.snapshot()
        snapshot.update({
            "max_steps": self.budget.max_steps,
            "max_retries": self.budget.max_retries,
            "max_duration_sec": self.budget.max_duration_sec,
            "max_risk_score": self.budget.max_risk_score,
        })
        return snapshot


__all__ = [
    "AutonomyBudget",
    "BudgetMonitor",
    "BudgetUsage",
    "DEFAULT_AUTONOMY_BUDGET",
    "MissionTermination",
    "SafetyContract",
]
