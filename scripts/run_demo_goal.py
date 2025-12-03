"""Run a full Planner v3 → AdaptiveController → BrowserWorkerV1 loop."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eikon_engine.pipelines.browser_pipeline import run_browser_goal

DEFAULT_GOAL = (
    "Log in to https://the-internet.herokuapp.com/login using credentials tomsmith / "
    "SuperSecretPassword! and capture a screenshot of the Secure Area."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute an autonomous browser goal demo.")
    parser.add_argument(
        "goal",
        nargs="?",
        default=DEFAULT_GOAL,
        help="Natural language instruction for the agent (defaults to Heroku login demo).",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=None,
        help="Optional path to settings.yaml (defaults to config/settings.yaml).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full orchestrator result as JSON after the summary.",
    )
    return parser.parse_args()


async def run(goal: str, settings_path: Path | None) -> Dict[str, Any]:
    """Delegate to the browser pipeline with the requested goal."""

    resolved = settings_path.resolve() if settings_path else None
    return await run_browser_goal(goal, settings_path=resolved)


def print_summary(result: Dict[str, Any]) -> None:
    completion = result.get("completion", {})
    artifacts = result.get("artifacts", {})
    goal = result.get("goal") or "goal"
    history = result.get("history") or []
    final_summary = result.get("final_summary") or {}
    steps_executed = len(final_summary.get("traces") or [])
    attempts = result.get("attempts", len(history))

    print("\n=== Demo Goal Complete ===")
    print(f"Goal: {goal}")
    print(f"Status: {'✅ complete' if completion.get('complete') else '⚠️ incomplete'}")
    print(f"Reason: {completion.get('reason', 'unknown')}")
    print(f"Attempts: {attempts}")
    print(f"Steps executed (last run): {steps_executed}")
    if artifacts:
        print("Artifacts:")
        print(f"  Base dir : {artifacts.get('base_dir')}")
        print(f"  Steps log: {artifacts.get('steps_file')}")
        print(f"  Trace log: {artifacts.get('trace_file')}")
        summary_file = Path(artifacts.get("base_dir", "")) / "run_summary.json"
        print(f"  Summary  : {summary_file}")
        print("Open the artifact directory to view screenshots and DOM snapshots.")


def main() -> None:
    args = parse_args()
    try:
        result = asyncio.run(run(args.goal, args.settings))
    except FileNotFoundError as exc:
        print(f"Configuration error: {exc}")
        raise SystemExit(1) from exc
    except KeyboardInterrupt:  # pragma: no cover - user cancellation
        print("Interrupted. Browser session stopped.")
        raise SystemExit(130)

    print_summary(result)
    if args.json:
        print("\nFull Result JSON:\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
