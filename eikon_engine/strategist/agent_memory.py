"""Agent-level episodic memory for selectors and subgoals."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentMemoryEntry:
    fingerprint: str
    selectors: List[str] = field(default_factory=list)
    subgoals: List[str] = field(default_factory=list)
    confidence_samples: List[float] = field(default_factory=list)
    reward_samples: List[float] = field(default_factory=list)
    behavior: Dict[str, Any] = field(default_factory=dict)


AgentMemoryHint = Dict[str, Any]


class AgentMemory:
    def __init__(self, *, max_entries: int = 50) -> None:
        self.max_entries = max_entries
        self._memory: "OrderedDict[str, AgentMemoryEntry]" = OrderedDict()

    def record(
        self,
        page_fingerprint: str,
        selector_repairs: List[str] | None,
        subgoal_usage: List[str] | None,
        reward_trace: List[Dict[str, Any]] | None,
        behavior_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = self._memory.get(page_fingerprint) or AgentMemoryEntry(fingerprint=page_fingerprint)
        if selector_repairs:
            entry.selectors = list({*entry.selectors, *selector_repairs})
        if subgoal_usage:
            entry.subgoals = list({*entry.subgoals, *subgoal_usage})
        if reward_trace:
            entry.reward_samples.extend(sample.get("reward", 0.0) for sample in reward_trace[-3:])
            entry.confidence_samples.extend((sample.get("confidence", {}) or {}).get("confidence", 0.0) for sample in reward_trace[-3:])
        if behavior_summary:
            entry.behavior = behavior_summary
            prediction = (behavior_summary.get("last_prediction") or {}) if isinstance(behavior_summary, dict) else {}
            recommended = prediction.get("recommended_subgoals") or []
            if recommended:
                entry.subgoals = list({*entry.subgoals, *recommended})
        self._memory[page_fingerprint] = entry
        self._memory.move_to_end(page_fingerprint)
        self._trim()

    def retrieve(self, page_fingerprint: str) -> Optional[AgentMemoryHint]:
        entry = self._memory.get(page_fingerprint)
        if not entry:
            return None
        self._memory.move_to_end(page_fingerprint)
        hint = {
            "selectors": entry.selectors,
            "subgoals": entry.subgoals,
            "avg_confidence": self._avg(entry.confidence_samples),
            "avg_reward": self._avg(entry.reward_samples),
        }
        if entry.behavior:
            hint["behavior"] = entry.behavior
            last_prediction = entry.behavior.get("last_prediction") or {}
            bias = last_prediction.get("selector_bias")
            if bias:
                hint["selector_bias"] = bias
            recommended = last_prediction.get("recommended_subgoals") or []
            if recommended:
                merged = list({*entry.subgoals, *recommended})
                hint["subgoals"] = merged
        return hint

    def summarize_experience(self) -> Dict[str, Any]:
        total = len(self._memory)
        avg_conf = self._avg([self._avg(entry.confidence_samples) for entry in self._memory.values()])
        difficulties = []
        for entry in self._memory.values():
            last_prediction = entry.behavior.get("last_prediction") if entry.behavior else None
            if last_prediction and isinstance(last_prediction.get("difficulty"), (int, float)):
                difficulties.append(last_prediction["difficulty"])
        return {
            "entries": total,
            "avg_confidence": avg_conf,
            "avg_difficulty": self._avg(difficulties),
        }

    def _trim(self) -> None:
        while len(self._memory) > self.max_entries:
            self._memory.popitem(last=False)

    def _avg(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)


__all__ = ["AgentMemory", "AgentMemoryHint", "AgentMemoryEntry"]
