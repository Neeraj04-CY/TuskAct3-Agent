from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from eikon_engine.pipelines.browser_pipeline import run_pipeline


def collect_interventions(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    interventions: List[Dict[str, Any]] = []
    for entry in trace or []:
        event = entry.get("event")
        if event == "insert":
            interventions.append(
                {
                    "type": entry.get("tag", "insert"),
                    "count": entry.get("count", 1),
                    "cursor": entry.get("cursor"),
                }
            )
        elif event in {"failure", "login_skip", "skip_rule"}:
            interventions.append(
                {
                    "type": event,
                    "detail": {k: v for k, v in entry.items() if k not in {"event"}},
                }
            )
    return interventions


def format_intervention(intervention: Dict[str, Any]) -> str:
    kind = intervention.get("type", "event")
    if kind == "cookie_popup":
        return f"Dismissed cookie banner ({intervention.get('count', 1)} step)"
    if kind == "micro_repair":
        return f"Patched selector ({intervention.get('count', 1)} step)"
    if kind == "failure":
        detail = intervention.get("detail", {})
        return f"Recorded failure signature: {detail.get('signature', 'unknown')}"
    if kind == "login_skip":
        detail = intervention.get("detail", {})
        return f"Skipped {detail.get('removed', 0)} login steps"
    if kind == "skip_rule":
        detail = intervention.get("detail", {})
        return f"Skip rule triggered: {detail.get('rule')}"
    return kind.replace("_", " ").title()


def build_summary(result: Dict[str, Any], artifact_dir: Path) -> Dict[str, Any]:
    completion = result.get("completion", {})
    run_context = result.get("run_context", {})
    interventions = collect_interventions(result.get("strategist_trace", []))
    summary = {
        "goal": result.get("goal"),
        "completed": completion.get("complete"),
        "completion_reason": completion.get("reason"),
        "step_count": result.get("step_count"),
        "duration_seconds": result.get("duration_seconds"),
        "interventions": interventions,
        "redirects": run_context.get("redirects", []),
        "failure_artifacts": run_context.get("failure_artifacts", []),
        "artifact_dir": str(artifact_dir),
    }
    return summary


def write_summary(summary: Dict[str, Any], *, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_summary(summary: Dict[str, Any]) -> None:
    print("Recovery Demo Summary")
    print("======================")
    print(f"Goal: {summary.get('goal')}")
    status = "success" if summary.get("completed") else "needs replan"
    print(f"Status: {status} ({summary.get('completion_reason')})")
    interventions = summary.get("interventions") or []
    if interventions:
        print("Interventions:")
        for item in interventions:
            print(f" - {format_intervention(item)}")
    else:
        print("Interventions: none detected")
    redirects = summary.get("redirects") or []
    if redirects:
        print("Redirects observed:")
        for redirect in redirects:
            print(f" - {redirect.get('from')} -> {redirect.get('to')}")
    artifacts_label = summary.get("artifact_dir")
    if artifacts_label:
        print(f"Artifacts stored in: {artifacts_label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the autonomous recovery demo and summarize interventions.")
    parser.add_argument(
        "goal",
        nargs="?",
        default="Navigate the recovery demo site, bypass blockers, and capture the dashboard",
        help="Goal description to feed into Strategist V2",
    )
    parser.add_argument("--execute", action="store_true", help="Run with a live Playwright browser instead of dry run")
    parser.add_argument("--allow-sensitive", action="store_true", help="Enable use of sensitive credentials if needed")
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional path to copy the recovery summary JSON (defaults to artifacts directory)",
    )
    args = parser.parse_args()

    result = run_pipeline(
        args.goal,
        allow_sensitive=args.allow_sensitive,
        dry_run=not args.execute,
        artifact_prefix="recovery_demo",
    )

    artifact_dir = Path(result.get("artifacts", {}).get("base_dir", Path("artifacts") / "recovery_demo_manual"))
    summary = build_summary(result, artifact_dir)
    summary_path = artifact_dir / "recovery_summary.json"
    write_summary(summary, path=summary_path)
    if args.summary:
        write_summary(summary, path=Path(args.summary))
    print_summary(summary)


if __name__ == "__main__":
    main()
