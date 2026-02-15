from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .signals import SkillSignal, load_skill_signals

UTC = timezone.utc


@dataclass(frozen=True)
class ImpactInput:
    skill: Optional[str]
    subgoal: str
    intent: str
    success_rate: float
    failure_rate: float
    confidence: float
    recency_seconds: Optional[float]


class LearningImpactScore:
    """Compute bounded learning impact scores in [-1.0, 1.0]."""

    def __init__(
        self,
        *,
        signals: Sequence[SkillSignal] | None = None,
        now: Optional[datetime] = None,
        recency_half_life: float = 3600.0,
        confidence_floor: float = 0.5,
    ) -> None:
        self.signals = list(signals or [])
        self.now = now or datetime.now(UTC)
        self.recency_half_life = max(1.0, float(recency_half_life))
        self.confidence_floor = float(confidence_floor)

    @classmethod
    def from_logs(cls, root: Path | str = Path("learning_logs"), *, min_confidence: float = 0.0) -> "LearningImpactScore":
        signals = load_skill_signals(root, min_confidence=min_confidence)
        return cls(signals=signals)

    def _match_signal(self, skill: Optional[str], intent: str) -> Optional[SkillSignal]:
        intent_lower = intent.lower()
        for signal in self.signals:
            if skill and signal.skill_name == skill:
                return signal
            if not skill and signal.mission_type in {intent_lower, "unknown"}:
                return signal
        return None

    def _recency_weight(self, timestamp: Optional[datetime]) -> float:
        if not timestamp:
            return 0.0
        delta = max((self.now - timestamp).total_seconds(), 0.0)
        # exponential decay; half-life controls steepness
        return exp(-delta / self.recency_half_life)

    def _build_input(self, skill: Optional[str], subgoal: str, intent: str) -> ImpactInput:
        signal = self._match_signal(skill, intent)
        if signal:
            success_rate = signal.success_rate
            failure_rate = max(0.0, 1.0 - signal.success_rate)
            confidence = signal.confidence_mean
            recency = self._recency_weight(signal.last_timestamp)
        else:
            success_rate = 0.0
            failure_rate = 0.0
            confidence = 0.5
            recency = None
        return ImpactInput(
            skill=skill,
            subgoal=subgoal,
            intent=intent,
            success_rate=success_rate,
            failure_rate=failure_rate,
            confidence=confidence,
            recency_seconds=recency,
        )

    def score(self, skill: Optional[str], subgoal: str, intent: str) -> float:
        payload = self._build_input(skill, subgoal, intent)
        base = (payload.success_rate - payload.failure_rate)
        if payload.confidence < self.confidence_floor:
            base -= (self.confidence_floor - payload.confidence) * 1.5
        if payload.recency_seconds is not None:
            base = base * (0.5 + payload.recency_seconds)
        # clamp to [-1, 1]
        return max(-1.0, min(1.0, round(base, 4)))

    def build_index(self) -> Dict[str, float]:
        index: Dict[str, float] = {}
        for signal in self.signals:
            key = self._index_key(signal.skill_name, signal.mission_type)
            index[key] = self.score(signal.skill_name, signal.mission_type, signal.mission_type)
        return index

    def _index_key(self, skill: Optional[str], intent: str) -> str:
        return f"{skill or 'unknown'}::{intent or 'unknown'}"

    def persist(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "scores": self.build_index(),
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target


def load_persisted_scores(path: Path | str) -> Dict[str, float]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        scores = payload.get("scores")
        return dict(scores) if isinstance(scores, dict) else {}
    except Exception:
        return {}


__all__ = ["LearningImpactScore", "load_persisted_scores"]
