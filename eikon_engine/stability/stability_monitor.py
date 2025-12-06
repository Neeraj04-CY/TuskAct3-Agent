"""Run-to-run stability monitoring utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
import difflib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


Number = float | int


@dataclass
class StabilityMonitor:
    """Tracks aggregate stability metrics and emits human-readable reports."""

    history_path: Path | str | None = None
    history_window: int = 50
    _history: List[Dict[str, Any]] = field(init=False, default_factory=list)
    _last_report: Optional[Dict[str, Any]] = field(init=False, default=None)

    def __post_init__(self) -> None:
        base_path = Path(self.history_path) if self.history_path else Path("artifacts") / "stability" / "history.json"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path = base_path
        self._history = self._load_history()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_run(
        self,
        *,
        goal: str,
        completion: Dict[str, Any] | None,
        run_context: Dict[str, Any] | None,
        strategist_trace: Sequence[Dict[str, Any]] | None = None,
        duration_seconds: Number | None = None,
        artifact_base: str | None = None,
    ) -> Dict[str, Any]:
        """Record metrics for a run and return the full stability report."""

        completion = completion or {}
        run_context = run_context or {}
        strategist_trace = list(strategist_trace or [])
        timestamp = datetime.now(UTC).isoformat()
        completed = bool(completion.get("complete"))
        reward_trace = run_context.get("reward_trace") or []
        repair_events = run_context.get("repair_events") or []
        duration = round(float(duration_seconds or 0.0), 3)
        avg_reward = self._avg(entry.get("reward", 0.0) for entry in reward_trace)
        previous_rewards = list(self._metric_values("avg_reward"))
        baseline_reward = self._avg(previous_rewards) if previous_rewards else avg_reward
        reward_drift = round(avg_reward - baseline_reward, 3)
        avg_confidence = self._avg(
            (entry.get("confidence") or {}).get("confidence", 0.0) for entry in reward_trace
        )
        previous_conf = list(self._metric_values("avg_confidence"))
        baseline_confidence = self._avg(previous_conf) if previous_conf else avg_confidence
        confidence_delta = round(avg_confidence - baseline_confidence, 3)
        repair_count = len(repair_events)
        previous_repairs = list(self._metric_values("repair_count"))
        baseline_repairs = self._avg(previous_repairs) if previous_repairs else float(repair_count)
        repair_delta = round(repair_count - baseline_repairs, 3)
        previous_durations = list(self._metric_values("duration_seconds"))
        baseline_duration = self._avg(previous_durations) if previous_durations else duration
        duration_delta = round(duration - baseline_duration, 3)
        failure_counter = self._collect_failures(run_context, strategist_trace)
        repeated_failures = {key: count for key, count in failure_counter.items() if count > 1}
        dom_fingerprint = run_context.get("current_fingerprint") or ""
        last_dom = self._history[-1]["metrics"].get("dom_fingerprint") if self._history else ""
        dom_similarity = round(self._fingerprint_similarity(dom_fingerprint, last_dom), 3)
        success_rate = self._compute_success_rate(completed)
        report = {
            "timestamp": timestamp,
            "goal": goal,
            "completed": completed,
            "artifact_base": artifact_base,
            "metrics": {
                "duration_seconds": duration,
                "avg_reward": round(avg_reward, 3),
                "reward_baseline": round(baseline_reward, 3),
                "reward_drift": reward_drift,
                "avg_confidence": round(avg_confidence, 3),
                "confidence_delta": confidence_delta,
                "repair_count": repair_count,
                "repair_delta": repair_delta,
                "duration_delta": duration_delta,
                "dom_fingerprint": dom_fingerprint,
                "dom_similarity_prev": dom_similarity,
                "success_rate": round(success_rate, 3),
                "repeated_failures": repeated_failures,
                "failure_reason": completion.get("reason") or "",
            },
            "trends": {
                "reward_trend": self._trend_label(reward_drift),
                "confidence_trend": self._trend_label(confidence_delta),
                "repair_trend": self._trend_label(repair_delta),
                "duration_trend": self._trend_label(duration_delta),
                "dom_similarity": dom_similarity,
            },
            "history_snapshot": self._history_snapshot(),
        }
        self._persist_entry(report)
        self._last_report = report
        return report

    def write_reports(self, report: Dict[str, Any], directory: Path | str) -> Dict[str, Path]:
        """Write JSON and Markdown reports to the provided directory."""

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "stability_report.json"
        md_path = output_dir / "stability_report.md"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return {"json": json_path, "markdown": md_path}

    def format_markdown(self, report: Dict[str, Any]) -> str:
        """Return a readable Markdown representation of the report."""

        metrics = report.get("metrics", {})
        trends = report.get("trends", {})
        lines: List[str] = []
        lines.append(f"# Stability Report — {report.get('timestamp')}")
        lines.append("")
        lines.append(f"- **Goal**: {report.get('goal')}")
        lines.append(f"- **Completed**: {'yes' if report.get('completed') else 'no'}")
        lines.append(f"- **Success rate (all runs)**: {metrics.get('success_rate', 0.0):.2f}")
        lines.append(f"- **Average reward**: {metrics.get('avg_reward', 0.0):.2f} (baseline {metrics.get('reward_baseline', 0.0):.2f})")
        lines.append(f"- **Average confidence**: {metrics.get('avg_confidence', 0.0):.2f}")
        lines.append(f"- **Repairs this run**: {metrics.get('repair_count', 0)}")
        lines.append(f"- **Duration (s)**: {metrics.get('duration_seconds', 0.0):.2f}")
        lines.append(f"- **DOM similarity to last run**: {trends.get('dom_similarity', 0.0):.2f}")
        lines.append("")
        lines.append("## Trends")
        lines.append(f"- Reward drift: {metrics.get('reward_drift', 0.0):+.3f} ({trends.get('reward_trend')})")
        lines.append(f"- Confidence delta: {metrics.get('confidence_delta', 0.0):+.3f} ({trends.get('confidence_trend')})")
        lines.append(f"- Repair trend: {metrics.get('repair_delta', 0.0):+.3f} ({trends.get('repair_trend')})")
        lines.append(f"- Duration trend: {metrics.get('duration_delta', 0.0):+.3f} ({trends.get('duration_trend')})")
        lines.append("")
        repeated = metrics.get("repeated_failures") or {}
        lines.append("## Repeated failures")
        if repeated:
            for reason, count in repeated.items():
                lines.append(f"- {reason}: {count}")
        else:
            lines.append("- None detected")
        lines.append("")
        snapshot = report.get("history_snapshot") or []
        if snapshot:
            lines.append("## Recent runs")
            lines.append("| Timestamp | Result | Reward | Confidence | Repairs | Duration |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for item in snapshot:
                metrics_block = item.get("metrics", {})
                lines.append(
                    f"| {item.get('timestamp')} | {'pass' if item.get('completed') else 'fail'} | "
                    f"{metrics_block.get('avg_reward', 0.0):.2f} | "
                    f"{metrics_block.get('avg_confidence', 0.0):.2f} | {metrics_block.get('repair_count', 0)} | "
                    f"{metrics_block.get('duration_seconds', 0.0):.2f} |"
                )
        else:
            lines.append("## Recent runs\nNo history available yet.")
        return "\n".join(lines)

    @property
    def last_report(self) -> Optional[Dict[str, Any]]:
        return self._last_report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_history(self) -> List[Dict[str, Any]]:
        if not self.history_path or not Path(self.history_path).exists():
            return []
        try:
            data = json.loads(Path(self.history_path).read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            return []
        return []

    def _save_history(self) -> None:
        Path(self.history_path).write_text(json.dumps(self._history, indent=2), encoding="utf-8")

    def _persist_entry(self, report: Dict[str, Any]) -> None:
        entry = {
            "timestamp": report.get("timestamp"),
            "goal": report.get("goal"),
            "completed": report.get("completed"),
            "metrics": {
                key: report.get("metrics", {}).get(key)
                for key in (
                    "avg_reward",
                    "avg_confidence",
                    "repair_count",
                    "duration_seconds",
                    "dom_fingerprint",
                )
            },
        }
        self._history.append(entry)
        if len(self._history) > self.history_window:
            self._history = self._history[-self.history_window :]
        self._save_history()

    def _history_snapshot(self, window: int = 5) -> List[Dict[str, Any]]:
        return self._history[-window:]

    def _metric_values(self, key: str) -> Iterable[float]:
        for entry in self._history:
            metrics = entry.get("metrics") or {}
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                yield float(value)

    def _avg(self, values: Iterable[Number]) -> float:
        total = 0.0
        count = 0
        for value in values:
            total += float(value)
            count += 1
        if count == 0:
            return 0.0
        return total / count

    def _compute_success_rate(self, completed: bool) -> float:
        success_count = sum(1 for entry in self._history if entry.get("completed"))
        total_runs = len(self._history)
        success_count += 1 if completed else 0
        total_runs += 1
        if total_runs == 0:
            return 0.0
        return success_count / total_runs

    def _collect_failures(
        self,
        run_context: Dict[str, Any],
        strategist_trace: Sequence[Dict[str, Any]],
    ) -> Counter[str]:
        reasons: List[str] = []
        for entry in run_context.get("history", []) or []:
            status = (entry.get("status") or "").strip()
            if status and status.lower() not in {"ok", "completed", "browser actions completed"}:
                reasons.append(status)
        for event in strategist_trace:
            label = event.get("event")
            if label in {"failure", "failure_detected"}:
                reason = event.get("reason") or event.get("signature") or label
                reasons.append(str(reason))
        return Counter(reasons)

    def _fingerprint_similarity(self, current: str, previous: str) -> float:
        if not current or not previous:
            return 0.0
        return difflib.SequenceMatcher(a=current, b=previous).ratio()

    def _trend_label(self, delta: float, tolerance: float = 0.05) -> str:
        if delta > tolerance:
            return "up"
        if delta < -tolerance:
            return "down"
        return "flat"


__all__ = ["StabilityMonitor"]
