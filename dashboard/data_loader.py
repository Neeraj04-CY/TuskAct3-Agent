from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List


def _artifact_root() -> Path:
    override = os.environ.get("EIKON_ARTIFACT_ROOT")
    return Path(override) if override else Path("artifacts")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_dashboard_payload() -> Dict[str, Any]:
    autonomy_root = _artifact_root() / "autonomy"
    latest_meta = _read_json(autonomy_root / "latest_run.json")
    run_path = latest_meta.get("run_path")
    if not run_path:
        return {"available": False, "message": "No autonomy run found"}
    run_dir = Path(run_path)
    if not run_dir.exists():
        return {"available": False, "message": "Run directory missing"}

    summary = _read_json(run_dir / "summary.json")
    result = _read_json(run_dir / "result.json")
    stability = _read_json(run_dir / "stability_report.json")

    run_ctx = result.get("run_context", {}) if isinstance(result, dict) else {}
    reward_trace = run_ctx.get("reward_trace", []) or []
    confidence_trace = _confidence_values(reward_trace)
    repair_events = run_ctx.get("repair_events", []) or []
    planner_evolution = run_ctx.get("plan_evolution", []) or []
    behavior_predictions = run_ctx.get("behavior_predictions", []) or []
    memory_summary = run_ctx.get("memory_summary", {}) or {}
    skill_events = (summary.get("skills") if isinstance(summary, dict) else None) or run_ctx.get("skills", []) or []
    skill_repairs = run_ctx.get("skill_repair_suggestions", []) or []
    stability_metrics = stability.get("metrics", {}) if isinstance(stability, dict) else {}
    stability_history = stability.get("history_snapshot", []) if isinstance(stability, dict) else []
    repeated_failures = stability_metrics.get("repeated_failures", {}) if isinstance(stability_metrics, dict) else {}
    dom_viewer = _extract_dom_assets(result.get("steps", []) if isinstance(result, dict) else [])

    payload = {
        "available": True,
        "summary": summary,
        "reward_trace": reward_trace,
        "confidence_trace": confidence_trace,
        "repair_events": repair_events,
        "planner_evolution": planner_evolution,
        "behavior_predictions": behavior_predictions,
        "memory_summary": memory_summary,
        "skills": skill_events,
        "skill_repairs": skill_repairs,
        "stability_metrics": stability_metrics,
        "stability_history": stability_history,
        "repeated_failures": _format_failure_clusters(repeated_failures),
        "dom_assets": dom_viewer,
    }
    return payload


def _confidence_values(reward_trace: List[Dict[str, Any]]) -> List[float]:
    values: List[float] = []
    for entry in reward_trace:
        confidence = entry.get("confidence") if isinstance(entry, dict) else None
        if isinstance(confidence, dict):
            val = confidence.get("confidence")
            if isinstance(val, (int, float)):
                values.append(float(val))
    return values


def _extract_dom_assets(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []
    for idx, entry in enumerate(steps, start=1):
        step_meta = entry.get("step", {}) if isinstance(entry, dict) else {}
        step_result = entry.get("result", {}) if isinstance(entry, dict) else {}
        screenshot_path = step_result.get("failure_screenshot_path") or step_result.get("screenshot_path")
        assets.append(
            {
                "step_id": step_meta.get("step_id") or f"step_{idx:02d}",
                "action": step_meta.get("action"),
                "dom_snapshot": step_result.get("dom_snapshot"),
                "screenshot_path": _normalize_artifact_uri(screenshot_path),
            }
        )
    return assets


def _format_failure_clusters(clusters: Dict[str, Any]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for reason, count in clusters.items():
        try:
            formatted.append({"reason": reason, "count": int(count)})
        except (TypeError, ValueError):
            continue
    return sorted(formatted, key=lambda item: item["count"], reverse=True)


def _normalize_artifact_uri(raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.exists():
        try:
            relative = candidate.resolve().relative_to(_artifact_root().resolve())
            return f"/artifacts/{relative.as_posix()}"
        except ValueError:
            return candidate.as_posix()
    return str(raw_path)


__all__ = ["load_dashboard_payload"]
