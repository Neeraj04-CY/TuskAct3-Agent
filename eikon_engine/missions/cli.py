"""Command-line entrypoint for mission execution."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from .mission_executor import MissionExecutor, run_mission_sync
from .mission_schema import MissionSpec

_TEST_EXECUTOR_ENV = "EIKON_MISSION_TEST_EXECUTOR"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mission orchestration CLI")
    parser.add_argument("--mission", required=True, help="Natural language mission instruction")
    parser.add_argument("--timeout", type=int, default=900, help="Mission timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries per subgoal")
    parser.add_argument("--execute", action="store_true", help="Use live Playwright (default dry-run)")
    parser.add_argument("--allow-sensitive", action="store_true", help="Allow sensitive context to reach planner")
    parser.add_argument(
        "--constraints",
        default=None,
        help="Optional JSON blob with mission constraints/context",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Root directory for mission artifacts",
    )
    parser.add_argument(
        "--debug-browser",
        action="store_true",
        help="Keep the Playwright browser open after mission completion",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    constraints = None
    if args.constraints:
        constraints = json.loads(args.constraints)
    spec = MissionSpec(
        instruction=args.mission,
        constraints=constraints,
        timeout_secs=args.timeout,
        max_retries=args.max_retries,
        allow_sensitive=args.allow_sensitive,
        execute=args.execute,
    )
    artifacts_dir = Path(args.artifacts_dir)
    executor = _resolve_executor(artifacts_dir, debug_browser=args.debug_browser)
    result = run_mission_sync(spec, executor=executor)
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0 if result.status == "complete" else 1


def _resolve_executor(artifacts_dir: Path, *, debug_browser: bool = False) -> MissionExecutor:
    hook = os.environ.get(_TEST_EXECUTOR_ENV)
    if hook:
        module_name, _, attr = hook.partition(":")
        module = importlib.import_module(module_name)
        factory: Callable[[Path], Any] = getattr(module, attr)
        executor = factory(artifacts_dir)
        if hasattr(executor, "debug_browser"):
            setattr(executor, "debug_browser", debug_browser)
        return executor
    return MissionExecutor(artifacts_root=artifacts_dir, debug_browser=debug_browser)


if __name__ == "__main__":  # pragma: no cover - exercised via CLI test
    sys.exit(main())
