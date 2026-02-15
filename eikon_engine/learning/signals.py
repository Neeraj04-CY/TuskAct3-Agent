from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .models import LearningRecord, LearningSkillUsage

UTC = timezone.utc


@dataclass(frozen=True)
class SkillSignal:
    """Aggregated evidence describing how well a skill performs for a mission type."""

    skill_name: str
    mission_type: str
    attempts: int
    successes: int
    total_steps_saved: int
    last_mission_id: Optional[str]
    last_timestamp: Optional[datetime]
    confidence_samples: int
    confidence_mean: float

    @property
    def success_rate(self) -> float:
        if not self.attempts:
            return 0.0
        return min(1.0, max(0.0, self.successes / self.attempts))

    @property
    def avg_steps_saved(self) -> float:
        if not self.attempts:
            return 0.0
        return self.total_steps_saved / self.attempts

    def score(self) -> float:
        """Priority score favoring high success rates and time savings."""

        bonus = min(self.avg_steps_saved / 5.0, 0.3)
        confidence_bonus = min(self.confidence_mean / 10.0, 0.1) if self.confidence_samples else 0.0
        return round(self.success_rate + bonus + confidence_bonus, 4)

    def as_dict(self) -> Dict[str, object]:
        return {
            "skill": self.skill_name,
            "mission_type": self.mission_type,
            "attempts": self.attempts,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 4),
            "avg_steps_saved": round(self.avg_steps_saved, 3),
            "total_steps_saved": self.total_steps_saved,
            "last_mission_id": self.last_mission_id,
            "last_observed_at": self.last_timestamp.isoformat() if self.last_timestamp else None,
            "confidence_mean": round(self.confidence_mean, 4),
            "confidence_samples": self.confidence_samples,
            "score": self.score(),
        }


@dataclass
class _SkillAccumulator:
    attempts: int = 0
    successes: int = 0
    total_steps_saved: int = 0
    last_timestamp: Optional[datetime] = None
    last_mission_id: Optional[str] = None
    confidence_sum: float = 0.0
    confidence_count: int = 0

    def update(self, *, success: bool, steps_saved: int, mission_id: str, timestamp: Optional[datetime], confidence: float) -> None:
        self.attempts += 1
        if success:
            self.successes += 1
        self.total_steps_saved += max(0, steps_saved)
        if timestamp and (self.last_timestamp is None or timestamp > self.last_timestamp):
            self.last_timestamp = timestamp
            self.last_mission_id = mission_id
        self.confidence_sum += max(0.0, confidence)
        self.confidence_count += 1

    def as_signal(self, *, skill_name: str, mission_type: str) -> SkillSignal:
        return SkillSignal(
            skill_name=skill_name,
            mission_type=mission_type,
            attempts=self.attempts,
            successes=self.successes,
            total_steps_saved=self.total_steps_saved,
            last_mission_id=self.last_mission_id,
            last_timestamp=self.last_timestamp,
            confidence_samples=self.confidence_count,
            confidence_mean=self.confidence_sum / self.confidence_count if self.confidence_count else 0.0,
        )


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _load_record(path: Path) -> Optional[LearningRecord]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - defensive parsing
        return None
    try:
        skills = [LearningSkillUsage(**entry) for entry in payload.get("skills_used", [])]
    except TypeError:
        skills = []
    return LearningRecord(
        mission_id=payload.get("mission_id", path.stem),
        timestamp=payload.get("timestamp"),
        mission_type=payload.get("mission_type", "unknown"),
        skills_used=skills,
        confidence_score=float(payload.get("confidence_score", 0.0) or 0.0),
        outcome=payload.get("outcome", "unknown"),
        trace_id=payload.get("trace_id"),
    )


def _iter_records(root: Path) -> Iterable[LearningRecord]:
    if not root.exists() or not root.is_dir():
        return []
    records: List[LearningRecord] = []
    for entry in sorted(root.glob("*.json")):
        record = _load_record(entry)
        if record:
            records.append(record)
    return records


def _bucket_key(record: LearningRecord, skill: LearningSkillUsage) -> Tuple[str, str]:
    mission_type = (record.mission_type or "unknown").lower()
    skill_name = skill.skill_name.strip() or "unknown"
    return mission_type, skill_name


def load_skill_signals(
    root: Path | str = Path("learning_logs"),
    *,
    min_confidence: float = 0.5,
) -> List[SkillSignal]:
    """Load aggregated skill signals from persisted learning logs."""

    min_confidence = max(0.0, min(1.0, min_confidence))
    root_path = Path(root)
    records = _iter_records(root_path)
    if not records:
        return []
    buckets: Dict[Tuple[str, str], _SkillAccumulator] = {}
    for record in records:
        if float(record.confidence_score or 0.0) < min_confidence:
            continue
        timestamp = _parse_timestamp(record.timestamp)
        for usage in record.skills_used:
            key = _bucket_key(record, usage)
            bucket = buckets.setdefault(key, _SkillAccumulator())
            bucket.update(
                success=bool(usage.success),
                steps_saved=int(usage.steps_saved or 0),
                mission_id=record.mission_id,
                timestamp=timestamp,
                confidence=float(record.confidence_score or 0.0),
            )
    signals = [bucket.as_signal(skill_name=key[1], mission_type=key[0]) for key, bucket in buckets.items() if bucket.attempts]
    signals.sort(key=lambda signal: (-signal.score(), -signal.attempts, signal.skill_name))
    return signals


__all__ = ["SkillSignal", "load_skill_signals"]
