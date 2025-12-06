from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable, Dict, List, Sequence


CurriculumBatch = Dict[str, Any]


def _avg(values: Sequence[float]) -> float:
    usable = [float(v) for v in values if isinstance(v, (int, float))]
    if not usable:
        return 0.0
    return float(mean(usable))


@dataclass
class _RunMetrics:
    difficulty: float = 0.5
    repeated_failures: int = 0
    reward_drift: float = 0.0
    stability_drift: float = 0.0
    dom_similarity: float = 0.0


class CurriculumBuilder:
    """Creates replay batches grouped by quality attributes."""

    def __init__(self, runs: Sequence[Dict[str, Any]]) -> None:
        self._runs = [dict(run) for run in runs]
        for run in self._runs:
            run["metrics"] = self._compute_metrics(run)

    def get_curriculum(self) -> List[CurriculumBatch]:
        remaining = list(self._runs)
        batches: List[CurriculumBatch] = []

        def pop_matching(predicate: Callable[[Dict[str, Any]], bool], tag: str, reason: str) -> None:
            matched: List[Dict[str, Any]] = []
            still: List[Dict[str, Any]] = []
            for run in remaining:
                if predicate(run):
                    matched.append(run)
                else:
                    still.append(run)
            if matched:
                batches.append({"tag": tag, "reason": reason, "runs": matched})
            remaining[:] = still

        pop_matching(lambda run: run["metrics"].difficulty >= 0.7, "high_difficulty", "Reward difficulty >= 0.7")
        pop_matching(lambda run: run["metrics"].repeated_failures > 0, "repeated_failures", "Runs with recorded failure clusters")
        pop_matching(lambda run: abs(run["metrics"].reward_drift) >= 0.1 or run["metrics"].stability_drift >= 0.1, "stability_drift", "Reward/confidence drift above 0.1")
        pop_matching(lambda run: run["metrics"].dom_similarity >= 0.7, "dom_similarity", "DOM fingerprints too similar to previous runs")
        if remaining:
            batches.append({"tag": "baseline", "reason": "All remaining runs", "runs": remaining})
        return batches

    def _compute_metrics(self, run: Dict[str, Any]) -> _RunMetrics:
        result = run.get("result", {}) or {}
        run_ctx = result.get("run_context", {}) if isinstance(result, dict) else {}
        stability = run.get("stability", {}) or {}
        metrics = stability.get("metrics", {}) if isinstance(stability, dict) else {}
        repeated_failures = metrics.get("repeated_failures") or {}
        difficulty = run_ctx.get("behavior_difficulty")
        if not isinstance(difficulty, (int, float)):
            behavior_predictions = run_ctx.get("behavior_predictions") or []
            difficulty = _avg(entry.get("difficulty") for entry in behavior_predictions)
        reward_drift = metrics.get("reward_drift") or 0.0
        stability_drift = metrics.get("confidence_delta") or 0.0
        dom_similarity = metrics.get("dom_similarity_prev") or 0.0
        return _RunMetrics(
            difficulty=float(difficulty or 0.5),
            repeated_failures=sum(int(val) for val in repeated_failures.values()) if isinstance(repeated_failures, dict) else 0,
            reward_drift=float(abs(reward_drift)),
            stability_drift=float(abs(stability_drift)),
            dom_similarity=float(dom_similarity or 0.0),
        )


__all__ = ["CurriculumBuilder", "CurriculumBatch"]
