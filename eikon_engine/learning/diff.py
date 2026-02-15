from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .signals import SkillSignal

UTC = timezone.utc


@dataclass(frozen=True)
class SkillSnapshot:
    skill: str
    mission_type: str
    priority: int
    confidence: float
    success_rate: float
    avg_steps_saved: float
    score: float
    attempts: int
    last_mission_id: Optional[str]
    last_observed_at: Optional[datetime]

    def as_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "priority": self.priority,
            "confidence": round(self.confidence, 4),
            "success_rate": round(self.success_rate, 4),
            "avg_steps_saved": round(self.avg_steps_saved, 3),
            "score": round(self.score, 4),
            "attempts": self.attempts,
        }
        if self.last_mission_id:
            payload["last_mission_id"] = self.last_mission_id
        if self.last_observed_at:
            payload["last_observed_at"] = self.last_observed_at.isoformat()
        return payload


def _key(signal: SkillSignal) -> Tuple[str, str]:
    mission_type = (signal.mission_type or "unknown").lower()
    return mission_type, signal.skill_name


def _rank_signals(signals: Sequence[SkillSignal]) -> Dict[Tuple[str, str], SkillSnapshot]:
    buckets: Dict[str, List[SkillSignal]] = {}
    for signal in signals:
        key = (signal.mission_type or "unknown").lower()
        bucket = buckets.setdefault(key, [])
        bucket.append(signal)
    snapshots: Dict[Tuple[str, str], SkillSnapshot] = {}
    for mission_type, bucket in buckets.items():
        bucket.sort(key=lambda sig: (-sig.score(), -sig.attempts, sig.skill_name))
        for index, signal in enumerate(bucket, start=1):
            snapshots[(mission_type, signal.skill_name)] = SkillSnapshot(
                skill=signal.skill_name,
                mission_type=mission_type,
                priority=index,
                confidence=signal.confidence_mean,
                success_rate=signal.success_rate,
                avg_steps_saved=signal.avg_steps_saved,
                score=signal.score(),
                attempts=signal.attempts,
                last_mission_id=signal.last_mission_id,
                last_observed_at=signal.last_timestamp,
            )
    return snapshots


def _avg_confidence(signals: Sequence[SkillSignal]) -> Optional[float]:
    if not signals:
        return None
    total = 0.0
    count = 0
    for signal in signals:
        total += max(0.0, signal.confidence_mean)
        count += 1
    if count == 0:
        return None
    return round(total / count, 4)


def _float_diff(a: Optional[float], b: Optional[float], *, epsilon: float = 1e-4) -> bool:
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    return abs(a - b) > epsilon


def _build_reason(before: Optional[SkillSnapshot], after: Optional[SkillSnapshot]) -> str:
    if before and after:
        changes: List[str] = []
        changes.append(f"success_rate {before.success_rate:.2f}->{after.success_rate:.2f}")
        if before.priority != after.priority:
            direction = "promoted" if after.priority < before.priority else "demoted"
            changes.append(f"priority {direction} {before.priority}->{after.priority}")
        if _float_diff(before.success_rate, after.success_rate):
            changes.append(f"success_rate {before.success_rate:.2f}->{after.success_rate:.2f}")
        if _float_diff(before.avg_steps_saved, after.avg_steps_saved):
            changes.append(f"avg_steps_saved {before.avg_steps_saved:.2f}->{after.avg_steps_saved:.2f}")
        if _float_diff(before.confidence, after.confidence):
            changes.append(f"confidence {before.confidence:.2f}->{after.confidence:.2f}")
        if not changes:
            changes.append(f"score {before.score:.2f}->{after.score:.2f}")
        if after.last_observed_at and (not before.last_observed_at or after.last_observed_at > before.last_observed_at):
            changes.append(f"last_used={after.last_observed_at.isoformat()}")
        return "; ".join(changes)
    if after:
        parts = [
            f"skill added with success_rate={after.success_rate:.2f}",
            f"confidence={after.confidence:.2f}",
        ]
        if after.last_observed_at:
            parts.append(f"last_used={after.last_observed_at.isoformat()}")
        return "; ".join(parts)
    return "skill removed after latest mission"


def _classify_change(before: Optional[SkillSnapshot], after: Optional[SkillSnapshot]) -> str:
    if before and after:
        if before.priority != after.priority:
            return "promotion" if after.priority < before.priority else "demotion"
        if _float_diff(before.score, after.score):
            return "metric_change"
        return "unchanged"
    if after:
        return "added"
    return "removed"


def build_skill_diff_report(
    mission_id: str,
    *,
    before_signals: Sequence[SkillSignal],
    after_signals: Sequence[SkillSignal],
) -> Dict[str, object]:
    before_map = _rank_signals(before_signals)
    after_map = _rank_signals(after_signals)
    skill_keys = sorted(set(before_map.keys()) | set(after_map.keys()))
    diffs: List[Dict[str, object]] = []
    for mission_type, skill in skill_keys:
        before = before_map.get((mission_type, skill))
        after = after_map.get((mission_type, skill))
        if before and after and before.as_dict() == after.as_dict():
            continue
        change_type = _classify_change(before, after)
        if change_type == "unchanged":
            continue
        diffs.append(
            {
                "skill": skill,
                "mission_type": mission_type,
                "before": before.as_dict() if before else None,
                "after": after.as_dict() if after else None,
                "change_type": change_type,
                "reason": _build_reason(before, after),
            }
        )
    avg_before = _avg_confidence(before_signals)
    avg_after = _avg_confidence(after_signals)
    report: Dict[str, object] = {
        "mission_id": mission_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "skill_diffs": diffs,
        "stats": {
            "before_avg_confidence": avg_before,
            "after_avg_confidence": avg_after,
            "before_skill_count": len(before_map),
            "after_skill_count": len(after_map),
        },
    }
    return report


def write_learning_diff_artifact(directory: Path | str, report: Dict[str, object]) -> Path:
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "learning_diff.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def build_learning_summary(report: Dict[str, object]) -> str:
    diffs: Sequence[Dict[str, object]] = report.get("skill_diffs", []) if isinstance(report, dict) else []
    stats = report.get("stats", {}) if isinstance(report, dict) else {}
    before_conf = stats.get("before_avg_confidence")
    after_conf = stats.get("after_avg_confidence")
    sentences: List[str] = []
    if diffs:
        sentences.append("Learning applied successfully.")
        promoted = sum(1 for entry in diffs if entry.get("change_type") == "promotion")
        demoted = sum(1 for entry in diffs if entry.get("change_type") == "demotion")
        added = sum(1 for entry in diffs if entry.get("change_type") == "added")
        removed = sum(1 for entry in diffs if entry.get("change_type") == "removed")
        highlight = diffs[0]
        before = highlight.get("before")
        after = highlight.get("after")
        if before and after and before.get("priority") and after.get("priority"):
            sentences.append(
                f"{highlight['skill']} priority {before['priority']}->{after['priority']} due to {highlight['reason']}."
            )
        elif after:
            sentences.append(f"{highlight['skill']} entered the ranking ({highlight['reason']}).")
        else:
            sentences.append(f"{highlight['skill']} dropped from the ranking.")
        if added:
            sentences.append(f"{added} skill{'s' if added != 1 else ''} gained enough signal to matter.")
        if removed:
            sentences.append(f"{removed} skill{'s' if removed != 1 else ''} lost relevance.")
        if demoted == 0:
            sentences.append("No skills were penalized.")
        elif promoted == 0:
            sentences.append("Skills were only demoted where metrics regressed.")
    else:
        sentences.append("Learning evaluated the mission but no skill priorities changed.")
    if before_conf is not None and after_conf is not None:
        if _float_diff(before_conf, after_conf):
            sentences.append(f"Learning confidence shifted {before_conf:.2f}->{after_conf:.2f}.")
        else:
            sentences.append(f"Learning confidence remained at {after_conf:.2f}.")
    return " ".join(sentences).strip()


def write_learning_summary(directory: Path | str, summary: str) -> Path:
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "learning_summary.txt"
    text = summary.strip() or "Learning evaluated the mission but produced no observable changes."
    path.write_text(text + "\n", encoding="utf-8")
    return path


def emit_learning_artifacts(
    *,
    mission_id: str,
    mission_dir: Path,
    before_signals: Sequence[SkillSignal],
    after_signals: Sequence[SkillSignal],
) -> Tuple[Path, Path]:
    report = build_skill_diff_report(mission_id, before_signals=before_signals, after_signals=after_signals)
    diff_path = write_learning_diff_artifact(mission_dir, report)
    summary_text = build_learning_summary(report)
    summary_path = write_learning_summary(mission_dir, summary_text)
    return diff_path, summary_path


__all__ = [
    "SkillSnapshot",
    "build_skill_diff_report",
    "write_learning_diff_artifact",
    "build_learning_summary",
    "write_learning_summary",
    "emit_learning_artifacts",
]
