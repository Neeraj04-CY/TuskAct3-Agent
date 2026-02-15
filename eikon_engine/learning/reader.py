from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import LearningRecord, LearningSkillUsage, LearningFailure


def _load_record(path: Path) -> Optional[LearningRecord]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    skills = [LearningSkillUsage(**entry) for entry in data.get("skills_used", [])]
    failures = [LearningFailure(**entry) for entry in data.get("failures", [])]
    return LearningRecord(
        mission_id=data.get("mission_id", path.stem),
        timestamp=data.get("timestamp"),
        mission_type=data.get("mission_type", "unknown"),
        skills_used=skills,
        failures=failures,
        confidence_score=float(data.get("confidence_score", 0.0)),
        outcome=data.get("outcome", "unknown"),
        trace_id=data.get("trace_id"),
    )


def _iter_records(root: Path) -> List[LearningRecord]:
    records: List[LearningRecord] = []
    for entry in sorted(root.glob("*.json")):
        record = _load_record(entry)
        if record:
            records.append(record)
    return records


def get_skill_stats(skill_name: str, *, root: Path | str = Path("learning_logs")) -> Dict[str, Any]:
    root_path = Path(root)
    records = _iter_records(root_path)
    attempts = 0
    successes = 0
    steps_saved = 0
    for record in records:
        for usage in record.skills_used:
            if usage.skill_name != skill_name:
                continue
            attempts += 1
            if usage.success:
                successes += 1
            steps_saved += usage.steps_saved
    return {
        "skill": skill_name,
        "attempts": attempts,
        "successes": successes,
        "steps_saved": steps_saved,
    }


def get_best_skill_for_intent(intent: str, *, root: Path | str = Path("learning_logs")) -> Optional[Dict[str, Any]]:
    root_path = Path(root)
    records = _iter_records(root_path)
    target = intent.lower()
    leaderboard: Dict[str, List[float]] = {}
    for record in records:
        if record.mission_type.lower() not in {target, "unknown"} and target not in record.mission_type.lower():
            continue
        for usage in record.skills_used:
            leaderboard.setdefault(usage.skill_name, []).append(1.0 if usage.success else 0.0)
    if not leaderboard:
        return None
    best_skill = None
    best_score = -1.0
    for name, values in leaderboard.items():
        score = sum(values) / len(values)
        if score > best_score:
            best_score = score
            best_skill = name
    if best_skill is None:
        return None
    return {"skill": best_skill, "success_rate": best_score}


def get_recent_success_patterns(mission_type: str, *, root: Path | str = Path("learning_logs"), limit: int = 10) -> List[Dict[str, Any]]:
    root_path = Path(root)
    records = _iter_records(root_path)
    target = mission_type.lower()
    patterns: List[Dict[str, Any]] = []
    for record in reversed(records):
        if len(patterns) >= limit:
            break
        if record.mission_type.lower() != target:
            continue
        patterns.append(
            {
                "mission_id": record.mission_id,
                "outcome": record.outcome,
                "confidence_score": record.confidence_score,
                "skills": [usage.skill_name for usage in record.skills_used if usage.success],
            }
        )
    return patterns


__all__ = [
    "get_skill_stats",
    "get_best_skill_for_intent",
    "get_recent_success_patterns",
]
