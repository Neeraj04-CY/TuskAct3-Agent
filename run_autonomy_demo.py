from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from eikon_engine.pipelines.browser_pipeline import run_pipeline


def collect_guardrail_blocks(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for entry in result.get("steps", []):
        worker_steps = entry.get("result", {}).get("steps", []) or []
        for step in worker_steps:
            if step.get("status") == "blocked":
                blocks.append({
                    "step_id": entry.get("step", {}).get("step_id"),
                    "action": entry.get("step", {}).get("action"),
                    "reason": step.get("block_reason"),
                })
    return blocks


def collect_interventions(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    interesting_events = {"insert", "progressive_recovery", "goal_chain_start", "goal_chain_queue", "failure"}
    return [entry for entry in trace if entry.get("event") in interesting_events]


def build_autonomy_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    completion = result.get("completion", {})
    run_ctx = result.get("run_context", {})
    summary = {
        "goal": result.get("goal"),
        "completed": completion.get("complete"),
        "reason": completion.get("reason"),
        "duration_seconds": result.get("duration_seconds"),
        "step_count": result.get("step_count"),
        "page_intents": run_ctx.get("page_intents", []),
        "redirects": run_ctx.get("redirects", []),
        "guardrail_blocks": collect_guardrail_blocks(result),
        "interventions": collect_interventions(result.get("strategist_trace", [])),
        "reward_trace": run_ctx.get("reward_trace", []),
        "repair_events": run_ctx.get("repair_events", []),
        "behavior_predictions": run_ctx.get("behavior_predictions", []),
    }
    return summary


def write_autonomy_summary(summary: Dict[str, Any], *, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_autonomy_summary(summary: Dict[str, Any], *, path: Path, debug_repair: bool = False) -> None:
    status = "success" if summary.get("completed") else "needs attention"
    print("Autonomy Demo Summary")
    print("======================")
    print(f"Goal: {summary.get('goal')}")
    print(f"Status: {status} ({summary.get('reason')})")
    intents = summary.get("page_intents") or []
    if intents:
        unique_intents = sorted({entry.get("intent") for entry in intents if entry.get("intent")})
        print(f"Observed intents: {', '.join(unique_intents)}")
    guardrail_blocks = summary.get("guardrail_blocks") or []
    if guardrail_blocks:
        print("Guardrail blocks:")
        for block in guardrail_blocks:
            print(f" - {block.get('reason')} on step {block.get('step_id')}")
    interventions = summary.get("interventions") or []
    if interventions:
        print(f"Interventions captured: {len(interventions)}")
    reward_trace = summary.get("reward_trace") or []
    if reward_trace:
        print("Step rewards:")
        for entry in reward_trace:
            conf = entry.get("confidence", {})
            print(f" - {entry.get('step_id')}: {entry.get('reward'):.2f} ({conf.get('band')})")
    repair_events = summary.get("repair_events") or []
    if repair_events:
        print(f"Repair events: {len(repair_events)}")
        if debug_repair:
            for event in repair_events:
                print(f"   * {event.get('patch', {}).get('type')}: {json.dumps(event)}")
    predictions = summary.get("behavior_predictions") or []
    if predictions:
        print("Behavior predictions:")
        for entry in predictions:
            diff = entry.get("difficulty")
            likelihood = "yes" if entry.get("likely_repair") else "no"
            bias = entry.get("selector_bias")
            step_id = entry.get("step_id") or entry.get("fingerprint", "unknown")
            diff_str = f"{diff:.2f}" if isinstance(diff, (int, float)) else "n/a"
            print(f" - {step_id}: difficulty={diff_str}, likely_repair={likelihood}, bias={bias}")
    print(f"Summary stored in: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the high-level autonomy demo and summarize interventions.")
    parser.add_argument(
        "goal",
        nargs="?",
        default="Demonstrate the full autonomy mode: reach the dashboard, dismiss blockers, and summarize findings.",
        help="Goal passed into Strategist V2",
    )
    parser.add_argument("--execute", action="store_true", help="Use a live Playwright session")
    parser.add_argument("--allow-sensitive", action="store_true", help="Allow sensitive resources when running live")
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional summary output path (defaults to artifacts/autonomy/autonomy_<timestamp>.json)",
    )
    parser.add_argument("--debug-repair", action="store_true", help="Print detailed self-repair patches")
    args = parser.parse_args()

    result = run_pipeline(
        args.goal,
        allow_sensitive=args.allow_sensitive,
        dry_run=not args.execute,
        artifact_prefix="autonomy_demo",
    )

    summary = build_autonomy_summary(result)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.summary or Path("artifacts") / "autonomy" / f"autonomy_{timestamp}.json")
    write_autonomy_summary(summary, path=output_path)
    print_autonomy_summary(summary, path=output_path, debug_repair=args.debug_repair)


if __name__ == "__main__":
    main()
