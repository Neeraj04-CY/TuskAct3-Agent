from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from eikon_engine.config_loader import load_settings
from eikon_engine.pipelines.browser_pipeline import PlannerV3Adapter, run_pipeline
from eikon_engine.stability import StabilityMonitor
from eikon_engine.strategist.strategist_v2 import StrategistV2

from run_autonomy_demo import build_autonomy_summary


EpisodePayload = Dict[str, Any]
EpisodeRunner = Callable[["EpisodeRequest"], EpisodePayload]


@dataclass(frozen=True)
class EpisodeRequest:
    index: int
    goal: str
    allow_sensitive: bool
    execute: bool


class RolloutEngine:
    """Runs multiple autonomy episodes and aggregates learning signals."""

    def __init__(
        self,
        *,
        goal: str,
        episodes: int,
        allow_sensitive: bool = False,
        execute: bool = False,
        output_root: Path | str | None = None,
        stability_monitor: StabilityMonitor | None = None,
        strategist: StrategistV2 | None = None,
        settings: Optional[Dict[str, Any]] = None,
        episode_runner: EpisodeRunner | None = None,
    ) -> None:
        if episodes < 1:
            raise ValueError("episodes must be >= 1")
        self.goal = goal
        self.episodes = episodes
        self.allow_sensitive = allow_sensitive
        self.execute = execute
        self.output_root = Path(output_root or Path("artifacts") / "rollouts")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.stability_monitor = stability_monitor or StabilityMonitor()
        self._episodes: List[Dict[str, Any]] = []
        if episode_runner is None:
            self.settings = settings or load_settings()
            planner_context = self.settings.get("planner", {})
            self.strategist = strategist or StrategistV2(planner=PlannerV3Adapter(context=planner_context))
            self._episode_runner = self._default_episode_runner
        else:
            self.settings = settings
            self.strategist = strategist
            self._episode_runner = episode_runner

    def run(self) -> Dict[str, Any]:
        for index in range(1, self.episodes + 1):
            request = EpisodeRequest(
                index=index,
                goal=self.goal,
                allow_sensitive=self.allow_sensitive,
                execute=self.execute,
            )
            payload = self._episode_runner(request)
            result = payload.get("result")
            if not result:
                raise RuntimeError("Episode runner must return a 'result' payload")
            summary = payload.get("summary") or build_autonomy_summary(result)
            run_dir = self.output_root / f"run_{index:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            stability_summary = result.get("stability") or self.stability_monitor.last_report
            if not stability_summary:
                raise RuntimeError("Stability summary missing from episode result")
            self._persist_episode(run_dir, result, summary, stability_summary)
            metrics = self._capture_episode_metrics(index, result, stability_summary)
            self._episodes.append(metrics)
        rollout_summary = self._build_rollout_summary()
        self._write_rollout_summary(rollout_summary)
        return rollout_summary

    # ------------------------------------------------------------------
    # Default episode runner (real autonomy pipeline)
    # ------------------------------------------------------------------
    def _default_episode_runner(self, request: EpisodeRequest) -> EpisodePayload:
        if self.settings is None or self.strategist is None:
            raise RuntimeError("RolloutEngine misconfigured: settings and strategist required")
        result = run_pipeline(
            request.goal,
            allow_sensitive=request.allow_sensitive,
            dry_run=not request.execute,
            settings=self.settings,
            artifact_prefix=f"rollout_run_{request.index:02d}",
            stability_monitor=self.stability_monitor,
            strategist=self.strategist,
        )
        summary = build_autonomy_summary(result)
        return {"result": result, "summary": summary}

    # ------------------------------------------------------------------
    # Episode persistence & metric capture
    # ------------------------------------------------------------------
    def _persist_episode(
        self,
        run_dir: Path,
        result: Dict[str, Any],
        summary: Dict[str, Any],
        stability: Dict[str, Any],
    ) -> None:
        (run_dir / "autonomy_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        signals = {
            "reward_trace": result.get("run_context", {}).get("reward_trace", []),
            "confidence_trace": self._confidence_trace(result.get("run_context", {})),
            "repair_events": result.get("run_context", {}).get("repair_events", []),
            "planner_evolution": result.get("run_context", {}).get("plan_evolution", []),
            "behavior_predictions": result.get("run_context", {}).get("behavior_predictions", []),
            "memory_summary": result.get("run_context", {}).get("memory_summary", {}),
            "dom_fingerprints": self._collect_dom_fingerprints(result.get("run_context", {})),
            "stability": stability,
        }
        (run_dir / "signals.json").write_text(json.dumps(signals, indent=2), encoding="utf-8")
        self.stability_monitor.write_reports(stability, run_dir)

    def _capture_episode_metrics(
        self,
        index: int,
        result: Dict[str, Any],
        stability: Dict[str, Any],
    ) -> Dict[str, Any]:
        run_ctx = result.get("run_context", {})
        reward_trace = run_ctx.get("reward_trace", [])
        confidence_trace = self._confidence_trace(run_ctx)
        planner_evolution = run_ctx.get("plan_evolution", [])
        behavior_predictions = run_ctx.get("behavior_predictions", [])
        repair_events = run_ctx.get("repair_events", [])
        dom_fingerprints = self._collect_dom_fingerprints(run_ctx)
        memory_summary = run_ctx.get("memory_summary", {})
        completion = result.get("completion", {})
        stability_metrics = stability.get("metrics", {})
        avg_reward = self._avg(entry.get("reward", 0.0) for entry in reward_trace)
        avg_difficulty = self._avg(
            entry.get("difficulty", 0.0) for entry in behavior_predictions if isinstance(entry.get("difficulty"), (int, float))
        )
        repair_likelihood = self._avg(1.0 if entry.get("likely_repair") else 0.0 for entry in behavior_predictions)
        return {
            "index": index,
            "avg_reward": avg_reward,
            "confidence_trace": confidence_trace,
            "confidence_median": self._median(confidence_trace),
            "repair_events": repair_events,
            "repair_count": len(repair_events),
            "planner_evolution": planner_evolution,
            "behavior_predictions": behavior_predictions,
            "avg_difficulty": avg_difficulty,
            "repair_likelihood": repair_likelihood,
            "stability_metrics": stability_metrics,
            "repeated_failures": stability_metrics.get("repeated_failures", {}),
            "memory_summary": memory_summary,
            "dom_fingerprints": dom_fingerprints,
            "completed": bool(completion.get("complete")),
            "completion_reason": completion.get("reason", ""),
            "duration_seconds": result.get("duration_seconds", 0.0),
        }

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
    def _build_rollout_summary(self) -> Dict[str, Any]:
        if not self._episodes:
            raise RuntimeError("No episodes recorded")
        avg_rewards = [entry["avg_reward"] for entry in self._episodes]
        confidence_medians = [entry["confidence_median"] for entry in self._episodes]
        repair_counts = [entry["repair_count"] for entry in self._episodes]
        difficulty_values = [entry["avg_difficulty"] for entry in self._episodes]
        repair_likelihoods = [entry["repair_likelihood"] for entry in self._episodes]
        stability_drifts = [entry["stability_metrics"].get("reward_drift", 0.0) for entry in self._episodes]
        successes = sum(1 for entry in self._episodes if entry["completed"])
        total = len(self._episodes)
        summary = {
            "goal": self.goal,
            "episodes": total,
            "successes": successes,
            "success_rate": successes / total,
            "reward_trend": self._linear_regression(avg_rewards),
            "confidence_medians": confidence_medians,
            "repair_trend": self._linear_regression(repair_counts),
            "repeated_failure_clusters": self._failure_clusters(),
            "behavior_model": {
                "difficulty_trend": self._linear_regression(difficulty_values),
                "repair_likelihood_trend": self._linear_regression(repair_likelihoods),
            },
            "memory_growth": self._memory_growth(),
            "stability_drift": {
                "scores": stability_drifts,
                "trend": self._linear_regression(stability_drifts),
            },
            "success_classification": [
                {"index": entry["index"], "completed": entry["completed"], "reason": entry["completion_reason"]}
                for entry in self._episodes
            ],
            "per_run": [
                {
                    "index": entry["index"],
                    "avg_reward": entry["avg_reward"],
                    "confidence_median": entry["confidence_median"],
                    "repairs": entry["repair_count"],
                    "avg_difficulty": entry["avg_difficulty"],
                    "duration_seconds": entry["duration_seconds"],
                    "completed": entry["completed"],
                }
                for entry in self._episodes
            ],
        }
        return summary

    def _write_rollout_summary(self, summary: Dict[str, Any]) -> None:
        json_path = self.output_root / "rollout_summary.json"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        md_path = self.output_root / "rollout_summary.md"
        md_path.write_text(self._format_markdown(summary), encoding="utf-8")

    def _format_markdown(self, summary: Dict[str, Any]) -> str:
        reward_trend = summary["reward_trend"]
        repair_trend = summary["repair_trend"]
        behavior_trend = summary["behavior_model"]["difficulty_trend"]
        stability_trend = summary["stability_drift"]["trend"]
        lines = [
            "# Rollout Summary",
            "",
            f"- Goal: {summary['goal']}",
            f"- Episodes: {summary['episodes']}",
            f"- Successes: {summary['successes']} ({summary['success_rate']:.0%})",
            f"- Reward trend slope: {reward_trend['slope']:.3f}",
            f"- Repair trend slope: {repair_trend['slope']:.3f}",
            f"- Behavior difficulty slope: {behavior_trend['slope']:.3f}",
            f"- Stability drift slope: {stability_trend['slope']:.3f}",
            "",
            "## Repeated failure clusters",
        ]
        clusters = summary["repeated_failure_clusters"]
        if clusters:
            lines.append("| Failure | Count |")
            lines.append("| --- | --- |")
            for entry in clusters:
                lines.append(f"| {entry['reason']} | {entry['count']} |")
        else:
            lines.append("No repeated failure clusters detected.")
        lines.append("")
        lines.append("## Per-run snapshot")
        lines.append("| Run | Reward | Confidence Median | Repairs | Difficulty | Result |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for entry in summary["per_run"]:
            result_label = "pass" if entry["completed"] else "fail"
            lines.append(
                f"| {entry['index']:02d} | {entry['avg_reward']:.3f} | {entry['confidence_median']:.3f} | "
                f"{entry['repairs']} | {entry['avg_difficulty']:.3f} | {result_label} |"
            )
        lines.append("")
        lines.append("## Stability drift")
        lines.append(", ".join(f"{score:.3f}" for score in summary["stability_drift"]["scores"]))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # math helpers
    # ------------------------------------------------------------------
    def _confidence_trace(self, run_ctx: Dict[str, Any]) -> List[float]:
        rewards = run_ctx.get("reward_trace", []) or []
        values: List[float] = []
        for entry in rewards:
            confidence = entry.get("confidence")
            if isinstance(confidence, dict):
                value = confidence.get("confidence")
                if isinstance(value, (int, float)):
                    values.append(float(value))
        return values

    def _collect_dom_fingerprints(self, run_ctx: Dict[str, Any]) -> List[str]:
        fingerprints = set()
        current = run_ctx.get("current_fingerprint")
        if isinstance(current, str) and current:
            fingerprints.add(current)
        for entry in run_ctx.get("behavior_predictions", []) or []:
            fp = entry.get("fingerprint")
            if isinstance(fp, str) and fp:
                fingerprints.add(fp)
        for hint in run_ctx.get("memory_hints", []) or []:
            fp = hint.get("fingerprint")
            if isinstance(fp, str) and fp:
                fingerprints.add(fp)
        return sorted(fingerprints)

    def _failure_clusters(self) -> List[Dict[str, Any]]:
        cluster: Dict[str, int] = {}
        for episode in self._episodes:
            repeated = episode.get("repeated_failures", {}) or {}
            for reason, count in repeated.items():
                cluster[reason] = cluster.get(reason, 0) + int(count)
        return [
            {"reason": reason, "count": cluster[reason]} for reason in sorted(cluster, key=lambda key: cluster[key], reverse=True)
        ]

    def _memory_growth(self) -> Dict[str, Any]:
        first = self._episodes[0].get("memory_summary") or {}
        last = self._episodes[-1].get("memory_summary") or {}
        return {
            "entries_start": first.get("entries", 0),
            "entries_end": last.get("entries", 0),
            "avg_confidence_start": first.get("avg_confidence", 0.0),
            "avg_confidence_end": last.get("avg_confidence", 0.0),
            "avg_difficulty_start": first.get("avg_difficulty", 0.0),
            "avg_difficulty_end": last.get("avg_difficulty", 0.0),
        }

    def _linear_regression(self, values: List[float]) -> Dict[str, float]:
        if not values:
            return {"slope": 0.0, "intercept": 0.0}
        if len(values) == 1:
            return {"slope": 0.0, "intercept": values[0]}
        xs = list(range(1, len(values) + 1))
        mean_x = sum(xs) / len(xs)
        mean_y = sum(values) / len(values)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
        denominator = sum((x - mean_x) ** 2 for x in xs)
        slope = numerator / denominator if denominator else 0.0
        intercept = mean_y - slope * mean_x
        return {"slope": slope, "intercept": intercept}

    def _avg(self, values: Any) -> float:
        total = 0.0
        count = 0
        for value in values:
            if isinstance(value, (int, float)):
                total += float(value)
                count += 1
        return total / count if count else 0.0

    def _median(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return float(statistics.median(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run N autonomy episodes and aggregate stability metrics.")
    parser.add_argument("goal", nargs="?", default="Demonstrate the full autonomy mode across multiple episodes.")
    parser.add_argument("--n", type=int, default=5, dest="episodes", help="Number of episodes to run")
    parser.add_argument("--execute", action="store_true", help="Use a live Playwright session")
    parser.add_argument("--allow-sensitive", action="store_true", help="Allow sensitive resources when running live")
    args = parser.parse_args()

    engine = RolloutEngine(
        goal=args.goal,
        episodes=args.episodes,
        allow_sensitive=args.allow_sensitive,
        execute=args.execute,
    )
    summary = engine.run()
    print(f"Rollout complete: {summary['successes']} / {summary['episodes']} successes.")
    print(f"Summary written to {engine.output_root / 'rollout_summary.json'}")


if __name__ == "__main__":
    main()
