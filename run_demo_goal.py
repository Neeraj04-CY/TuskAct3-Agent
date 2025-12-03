from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from eikon_engine.pipelines.browser_pipeline import run_pipeline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_demo")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Auto EIKON Strategist V2 demo goal.")
    parser.add_argument(
        "goal",
        nargs="?",
        default="Log in to HerokuApp demo and screenshot the dashboard",
        help="Goal text to execute",
    )
    parser.add_argument("--execute", action="store_true", help="Run with live browser (requires Playwright)")
    parser.add_argument("--allow-sensitive", action="store_true", help="Allow using sensitive credentials or data")
    parser.add_argument("--out", default="artifacts/demo_run.json", help="Where to write run JSON summary")
    args = parser.parse_args()

    result = run_pipeline(
        args.goal,
        allow_sensitive=args.allow_sensitive,
        dry_run=not args.execute,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote run summary to %s", out_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
