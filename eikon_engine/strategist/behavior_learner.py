"""Behavioral learner for strategist predictions."""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


Prediction = Dict[str, Any]
PlannerEvent = Dict[str, Any]


@dataclass
class BehaviorStats:
    reward_history: List[float] = field(default_factory=list)
    total_repairs: int = 0
    total_interference: int = 0
    selector_failures: Dict[str, int] = field(default_factory=lambda: {"css": 0, "text": 0, "role": 0})
    subgoal_counts: Dict[str, int] = field(default_factory=dict)
    subgoal_success: Dict[str, int] = field(default_factory=dict)
    total_steps: int = 0
    episodes: int = 0
    last_prediction: Prediction = field(default_factory=dict)


class BehaviorLearner:
    def __init__(self, *, reward_window: int = 30) -> None:
        self.reward_window = reward_window
        self._stats: Dict[str, BehaviorStats] = {}

    def predict(self, dom_fingerprint: str, recent_rewards: Optional[List[float]], repair_history: Optional[List[Dict[str, Any]]]) -> Prediction:
        stats = self._stats.setdefault(dom_fingerprint, BehaviorStats())
        rewards = (recent_rewards or []) or stats.reward_history[-self.reward_window :]
        avg_reward = self._avg(rewards)
        repair_count = len(repair_history or [])
        repair_rate = stats.total_repairs / max(1, stats.total_steps)
        selector_fail_rate = sum(stats.selector_failures.values()) / max(1, stats.total_steps)
        interference_rate = stats.total_interference / max(1, stats.total_steps)
        difficulty = self._clamp(0.5 - 0.3 * avg_reward + 0.2 * repair_rate + 0.1 * selector_fail_rate + 0.1 * interference_rate)
        likely_repair = difficulty >= 0.65 or repair_count >= 2 or repair_rate >= 0.25
        selector_bias = self._infer_selector_bias(stats)
        recommended_subgoals = self._recommend_subgoals(stats)
        prediction = {
            "difficulty": round(difficulty, 3),
            "likely_repair": likely_repair,
            "recommended_subgoals": recommended_subgoals,
            "selector_bias": selector_bias,
        }
        stats.last_prediction = prediction
        self._stats[dom_fingerprint] = stats
        return prediction

    def update(
        self,
        dom_fingerprint: str,
        reward_trace: Optional[List[Dict[str, Any]]],
        planner_events: Optional[List[PlannerEvent]],
        repair_events: Optional[List[Dict[str, Any]]],
    ) -> None:
        stats = self._stats.setdefault(dom_fingerprint, BehaviorStats())
        new_rewards = [entry.get("reward", 0.0) for entry in (reward_trace or [])]
        if new_rewards:
            stats.reward_history = (stats.reward_history + new_rewards)[-self.reward_window :]
            stats.total_steps += len(new_rewards)
        if repair_events:
            stats.total_repairs += len(repair_events)
            for event in repair_events:
                style = self._infer_selector_style(event)
                stats.selector_failures[style] = stats.selector_failures.get(style, 0) + 1
                if self._is_interference(event):
                    stats.total_interference += 1
        if planner_events:
            for event in planner_events:
                if event.get("type") != "subgoal":
                    continue
                name = self._normalize_subgoal_name(event.get("name"))
                stats.subgoal_counts[name] = stats.subgoal_counts.get(name, 0) + 1
                if event.get("status") == "completed":
                    stats.subgoal_success[name] = stats.subgoal_success.get(name, 0) + 1
        stats.episodes += 1
        self._stats[dom_fingerprint] = stats

    def summarize(self) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for fingerprint, stats in self._stats.items():
            summary[fingerprint] = {
                "avg_reward": self._avg(stats.reward_history),
                "repairs": stats.total_repairs,
                "interference": stats.total_interference,
                "selector_failures": dict(stats.selector_failures),
                "subgoals": self._summarize_subgoals(stats),
                "episodes": stats.episodes,
                "last_prediction": stats.last_prediction,
            }
        return summary

    def _summarize_subgoals(self, stats: BehaviorStats) -> Dict[str, Any]:
        details: Dict[str, Any] = {}
        for name, count in stats.subgoal_counts.items():
            success = stats.subgoal_success.get(name, 0)
            details[name] = {
                "count": count,
                "success_rate": success / count if count else 0.0,
            }
        return details

    def _recommend_subgoals(self, stats: BehaviorStats) -> List[str]:
        scored = sorted(stats.subgoal_counts.items(), key=lambda item: item[1], reverse=True)
        recommendations: List[str] = []
        for name, count in scored:
            if count < 2:
                continue
            success = stats.subgoal_success.get(name, 0)
            if count - success >= 1:
                recommendations.append(name)
            if len(recommendations) >= 3:
                break
        return recommendations

    def _infer_selector_bias(self, stats: BehaviorStats) -> str:
        failure_items = sorted(stats.selector_failures.items(), key=lambda item: item[1])
        if not failure_items:
            return "css"
        return failure_items[0][0]

    def _infer_selector_style(self, event: Dict[str, Any]) -> str:
        reason = (event.get("patch") or {}).get("reason") or event.get("reason", "")
        reason_lower = reason.lower()
        if "label" in reason_lower or "text" in reason_lower:
            return "text"
        if "role" in reason_lower:
            return "role"
        if "aria" in reason_lower:
            return "role"
        return "css"

    def _is_interference(self, event: Dict[str, Any]) -> bool:
        reason = (event.get("patch") or {}).get("reason") or event.get("reason", "")
        return "interference" in reason.lower()

    def _avg(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _clamp(self, value: float, *, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    def _normalize_subgoal_name(self, value: Any) -> str:
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            nested = value.get("name")
            if isinstance(nested, str) and nested:
                return nested
            return json.dumps(value, sort_keys=True, default=str)
        if value is None:
            return "unknown"
        return str(value)


__all__ = ["BehaviorLearner", "Prediction", "PlannerEvent"]
