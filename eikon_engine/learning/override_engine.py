from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PlannerConflict:
    planner_step: Dict[str, Any]
    learning_score: float
    historical_failures: int
    recommendation: str


@dataclass(frozen=True)
class OverrideDecision:
    decision_type: str
    reason: str
    confidence: float
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    adjusted_plan: Optional[List[Dict[str, Any]]] = None


class LearningOverrideEngine:
    def __init__(
        self,
        *,
        scores: Dict[Tuple[str, str, str], float] | None = None,
        preferred_skills: Sequence[str] | None = None,
        threshold: float = 0.0,
        hard_floor: float = -0.6,
    ) -> None:
        self.scores = dict(scores or {})
        self.preferred_skills = list(preferred_skills or [])
        self.threshold = float(threshold)
        self.hard_floor = float(hard_floor)

    def _score_for(self, step: Dict[str, Any]) -> float:
        skill = step.get("skill")
        subgoal = (step.get("description") or "").lower()
        intent = (step.get("intent") or step.get("bucket") or "unknown").lower()
        key = (skill or "unknown", subgoal, intent)
        return float(self.scores.get(key, self.scores.get(("unknown", subgoal, intent), 0.0)))

    def _evidence(self, plan: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for step in plan:
            payload.append({"step": step, "score": self._score_for(step)})
        return payload

    def apply_override(self, plan: Sequence[Dict[str, Any]], learning_context: Dict[str, Any]) -> OverrideDecision:
        evidence = self._evidence(plan)
        worst = min(evidence, key=lambda entry: entry["score"], default=None)
        if worst and worst["score"] <= self.hard_floor:
            return OverrideDecision(
                decision_type="REFUSE",
                reason="learning_score_below_hard_floor",
                confidence=abs(worst["score"]),
                evidence=evidence,
                adjusted_plan=None,
            )
        adjusted: List[Dict[str, Any]] = [dict(step) for step in plan]
        replaced = False
        reordered = False
        for step in adjusted:
            score = self._score_for(step)
            if score < self.threshold:
                if self.preferred_skills:
                    step["skill"] = self.preferred_skills[0]
                    replaced = True
                else:
                    step["skip_reason"] = "low_learning_score"
        # remove skipped
        filtered = [step for step in adjusted if not step.get("skip_reason")]
        if filtered and len(filtered) != len(adjusted):
            adjusted = filtered
        # reorder by descending score if multiple
        if len(adjusted) > 1 and not replaced:
            reordered = True
            adjusted.sort(key=lambda step: self._score_for(step), reverse=True)
        if replaced:
            return OverrideDecision(
                decision_type="REPLACE_WITH_SKILL",
                reason="low_score_replaced_with_preferred_skill",
                confidence=1.0,
                evidence=evidence,
                adjusted_plan=adjusted,
            )
        if reordered:
            return OverrideDecision(
                decision_type="REORDER",
                reason="reordered_by_learning_score",
                confidence=0.9,
                evidence=evidence,
                adjusted_plan=adjusted,
            )
        if any(entry["score"] < self.threshold for entry in evidence):
            return OverrideDecision(
                decision_type="SKIP",
                reason="low_score_step_skipped",
                confidence=0.6,
                evidence=evidence,
                adjusted_plan=adjusted,
            )
        return OverrideDecision(
            decision_type="ACCEPT",
            reason="learning_scores_ok",
            confidence=0.8,
            evidence=evidence,
            adjusted_plan=adjusted,
        )


__all__ = ["LearningOverrideEngine", "PlannerConflict", "OverrideDecision"]
