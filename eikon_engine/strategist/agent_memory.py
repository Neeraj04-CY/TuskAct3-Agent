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
    stability: Dict[str, Any] = field(default_factory=dict)


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
        stability_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = self._memory.get(page_fingerprint) or AgentMemoryEntry(fingerprint=page_fingerprint)
        if selector_repairs:
            entry.selectors = self._merge_unique(entry.selectors, selector_repairs)
        if subgoal_usage:
            for sg in subgoal_usage:
                if sg not in entry.subgoals:
                    entry.subgoals.append(sg)
        if reward_trace:
            entry.reward_samples.extend(sample.get("reward", 0.0) for sample in reward_trace[-3:])
            entry.confidence_samples.extend((sample.get("confidence", {}) or {}).get("confidence", 0.0) for sample in reward_trace[-3:])
        if behavior_summary:
            entry.behavior = behavior_summary
            prediction = (behavior_summary.get("last_prediction") or {}) if isinstance(behavior_summary, dict) else {}
            recommended = prediction.get("recommended_subgoals") or []
            if recommended:
                for sg in recommended:
                    if sg not in entry.subgoals:
                        entry.subgoals.append(sg)
        if stability_summary:
            entry.stability = stability_summary
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
                hint["subgoals"] = self._merge_unique(entry.subgoals, recommended)
        if entry.stability:
            hint["stability"] = entry.stability
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

    def store_stability(self, page_fingerprint: str, stability_summary: Dict[str, Any]) -> None:
        entry = self._memory.get(page_fingerprint) or AgentMemoryEntry(fingerprint=page_fingerprint)
        entry.stability = stability_summary
        self._memory[page_fingerprint] = entry
        self._memory.move_to_end(page_fingerprint)
        self._trim()

    def export(self) -> List[Dict[str, Any]]:
        exported: List[Dict[str, Any]] = []
        for entry in self._memory.values():
            exported.append({
                "fingerprint": entry.fingerprint,
                "selectors": list(entry.selectors),
                "subgoals": list(entry.subgoals),
                "avg_confidence": self._avg(entry.confidence_samples),
                "avg_reward": self._avg(entry.reward_samples),
                "behavior": dict(entry.behavior),
                "stability": dict(entry.stability),
            })
        return exported

    def _trim(self) -> None:
        while len(self._memory) > self.max_entries:
            self._memory.popitem(last=False)

    def _avg(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _merge_unique(self, existing: List[Any], additions: List[Any]) -> List[Any]:
        merged = list(existing)
        seen = {self._hashable_key(item) for item in merged}
        for item in additions:
            key = self._hashable_key(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

    def _hashable_key(self, value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(sorted((k, self._hashable_key(v)) for k, v in value.items()))
        if isinstance(value, list):
            return tuple(self._hashable_key(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted(self._hashable_key(v) for v in value))
        return value


__all__ = ["AgentMemory", "AgentMemoryHint", "AgentMemoryEntry"]
