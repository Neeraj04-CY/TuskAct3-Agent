from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .signals import SkillSignal, load_skill_signals


def infer_mission_type(text: Optional[str]) -> str:
    """Match mission instructions to the coarse buckets used by learning logs."""

    if not text:
        return "unknown"
    lowered = text.lower()
    if "login" in lowered:
        return "login"
    if any(token in lowered for token in ("list", "listing")):
        return "listing"
    if any(token in lowered for token in ("extract", "scrape", "harvest")):
        return "extraction"
    if "dashboard" in lowered:
        return "dashboard"
    return "unknown"


@dataclass(frozen=True)
class LearningBias:
    mission_type: str
    preferred_skills: List[str]
    signals: List[SkillSignal]
    source: str = "learning_index"

    def as_metadata(self, context: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "source": self.source,
            "mission_type": self.mission_type,
            "preferred_skills": list(self.preferred_skills),
            "signals": [signal.as_dict() for signal in self.signals],
        }
        if context:
            payload["context"] = dict(context)
        return payload

    def metadata_for(self, skill_name: str) -> Optional[Dict[str, object]]:
        for signal in self.signals:
            if signal.skill_name == skill_name:
                return {
                    "source": self.source,
                    "mission_type": self.mission_type,
                    "signal": signal.as_dict(),
                }
        return None


class LearningIndex:
    def __init__(self, *, signals: Sequence[SkillSignal]) -> None:
        self._signals = list(signals)
        self._by_type: Dict[str, List[SkillSignal]] = {}
        for signal in self._signals:
            mission_type = signal.mission_type or "unknown"
            bucket = self._by_type.setdefault(mission_type, [])
            bucket.append(signal)
        for bucket in self._by_type.values():
            bucket.sort(key=lambda signal: (-signal.score(), -signal.attempts, signal.skill_name))

    @classmethod
    def from_logs(
        cls,
        *,
        root: Path | str = Path("learning_logs"),
        min_confidence: float = 0.5,
    ) -> "LearningIndex":
        signals = load_skill_signals(root, min_confidence=min_confidence)
        return cls(signals=signals)

    @property
    def is_empty(self) -> bool:
        return not self._signals

    def top_signals(self, mission_type: Optional[str], *, limit: int = 3) -> List[SkillSignal]:
        target = (mission_type or "unknown").lower()
        matches = self._by_type.get(target)
        if not matches and target != "unknown":
            matches = self._by_type.get("unknown", [])
        if not matches:
            return []
        return list(matches[:limit])

    def build_bias(self, *, instruction: Optional[str], limit: int = 3) -> Optional[LearningBias]:
        mission_type = infer_mission_type(instruction)
        signals = self.top_signals(mission_type, limit=limit)
        if not signals and mission_type != "unknown":
            signals = self.top_signals("unknown", limit=limit)
            if signals:
                mission_type = "unknown"
        if not signals:
            return None
        preferred = [signal.skill_name for signal in signals]
        return LearningBias(mission_type=mission_type, preferred_skills=preferred, signals=list(signals))


class LearningIndexCache:
    def __init__(
        self,
        *,
        root: Path | str = Path("learning_logs"),
        min_confidence: float = 0.5,
    ) -> None:
        self.root = Path(root)
        self.min_confidence = float(min_confidence)
        self._snapshot: Tuple[Tuple[str, int, int], ...] = ()
        self._index: LearningIndex | None = None

    def _snapshot_paths(self) -> Tuple[Tuple[str, int, int], ...]:
        if not self.root.exists():
            return ()
        entries: List[Tuple[str, int, int]] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((path.name, int(stat.st_mtime_ns), stat.st_size))
        return tuple(entries)

    def _is_stale(self) -> bool:
        return self._snapshot != self._snapshot_paths()

    def get_index(self) -> LearningIndex:
        if self._index is None or self._is_stale():
            self._index = LearningIndex.from_logs(root=self.root, min_confidence=self.min_confidence)
            self._snapshot = self._snapshot_paths()
        return self._index

    def bias_for_goal(self, instruction: Optional[str], *, limit: int = 3) -> Optional[LearningBias]:
        index = self.get_index()
        if index.is_empty:
            return None
        return index.build_bias(instruction=instruction, limit=limit)


__all__ = ["LearningBias", "LearningIndex", "LearningIndexCache", "infer_mission_type"]
