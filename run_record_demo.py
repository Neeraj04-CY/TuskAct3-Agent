from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from eikon_engine.pipelines.browser_pipeline import run_pipeline


def build_record_dataset(result: Dict[str, Any], *, inline_dom: bool = False) -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []
    for entry in result.get("steps", []):
        step_meta = entry.get("step", {})
        worker_result = entry.get("result", {})
        action_payload = step_meta.get("action_payload", {})
        completion = worker_result.get("completion", {})
        record = {
            "step_id": step_meta.get("step_id"),
            "task_id": step_meta.get("task_id"),
            "action": action_payload.get("action"),
            "selector": action_payload.get("selector"),
            "url": action_payload.get("url"),
            "status": completion.get("reason"),
            "complete": completion.get("complete"),
            "error": worker_result.get("error"),
            "blocked": any(step.get("status") == "blocked" for step in worker_result.get("steps", [])) if isinstance(worker_result.get("steps"), list) else False,
        }
        dom_snapshot = worker_result.get("dom_snapshot")
        if inline_dom and dom_snapshot:
            record["dom_snapshot"] = dom_snapshot
        else:
            record["dom_path"] = worker_result.get("failure_dom_path")
        record["screenshot"] = worker_result.get("failure_screenshot_path")
        dataset.append(record)
    return dataset


def write_record_dataset(records: List[Dict[str, Any]], *, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_record_summary(records: List[Dict[str, Any]], *, output_path: Path) -> None:
    print("Recorded Demo Dataset")
    print("====================")
    print(f"Entries: {len(records)}")
    blocked = sum(1 for record in records if record.get("blocked"))
    print(f"Blocked steps: {blocked}")
    print(f"Output file: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Strategist V2 step data for offline analysis.")
    parser.add_argument(
        "goal",
        nargs="?",
        default="Traverse the demo site, collect artifacts, and summarize what happened.",
        help="Goal to feed into the pipeline.",
    )
    parser.add_argument("--execute", action="store_true", help="Use live Playwright execution instead of dry-run")
    parser.add_argument("--inline-dom", action="store_true", help="Embed DOM snapshots directly in the dataset")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSONL output path (defaults to artifacts/recordings/record_<timestamp>.jsonl)",
    )
    args = parser.parse_args()

    result = run_pipeline(
        args.goal,
        allow_sensitive=False,
        dry_run=not args.execute,
        artifact_prefix="record_demo",
    )

    records = build_record_dataset(result, inline_dom=args.inline_dom)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output or Path("artifacts") / "recordings" / f"record_{timestamp}.jsonl")
    write_record_dataset(records, path=output_path)
    print_record_summary(records, output_path=output_path)


if __name__ == "__main__":
    main()
