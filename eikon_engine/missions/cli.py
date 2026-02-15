"""Command-line entrypoint for mission execution."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict

from eikon_engine.config_loader import load_settings

from .mission_executor import MissionExecutor, run_mission_sync
from .mission_schema import MissionSpec

_TEST_EXECUTOR_ENV = "EIKON_MISSION_TEST_EXECUTOR"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mission orchestration CLI")
    parser.add_argument("--mission", help="Natural language mission instruction")
    parser.add_argument("--timeout", type=int, default=900, help="Mission timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries per subgoal")
    parser.add_argument("--execute", action="store_true", help="Use live Playwright (default dry-run)")
    parser.add_argument("--allow-sensitive", action="store_true", help="Allow sensitive context to reach planner")
    parser.add_argument("--autonomy-budget", default=None, help="JSON blob overriding the entire autonomy budget")
    parser.add_argument("--budget-max-steps", type=int, default=None, help="Override autonomy budget max steps")
    parser.add_argument("--budget-max-retries", type=int, default=None, help="Override autonomy budget max retries")
    parser.add_argument("--budget-max-duration", type=float, default=None, help="Override autonomy budget max duration (seconds)")
    parser.add_argument("--budget-max-risk", type=float, default=None, help="Override autonomy budget max risk score")
    parser.add_argument("--safety-contract", default=None, help="JSON blob describing allowed/blocked actions")
    parser.add_argument("--ask-on-uncertainty", action="store_true", help="Escalate to ASK when confidence stays low")
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
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in fast demo mode (less waiting, skip retries)",
    )
    parser.add_argument(
        "--demo-force-actions",
        action="store_true",
        help="Force a visible demo preflight (search + scroll) before planner/judgment",
    )
    parser.add_argument(
        "--canonical",
        default=None,
        help="Canonical mission slug defined in config/canonical_missions.json",
    )
    parser.add_argument(
        "--canonical-file",
        default="config/canonical_missions.json",
        help="Path to the canonical mission manifest",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from checkpoint file path or mission id",
    )
    args = parser.parse_args(argv)
    if not args.resume and not args.mission and not args.canonical:
        parser.error("Provide --mission text or choose a --canonical mission slug")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    canonical_overrides: Dict[str, Any] = {}
    if args.canonical:
        canonical_overrides = _load_canonical_spec(args.canonical, Path(args.canonical_file))
    instruction = canonical_overrides.get("instruction") or args.mission or "resume-mission"
    if not instruction:
        raise ValueError("Mission instruction missing; provide --mission text or a canonical slug")
    constraints = canonical_overrides.get("constraints")
    if args.constraints:
        constraints = json.loads(args.constraints)
    if args.demo_force_actions:
        constraints = dict(constraints or {})
        constraints["demo_force_actions"] = True
    timeout = canonical_overrides.get("timeout_secs", args.timeout)
    max_retries = canonical_overrides.get("max_retries", args.max_retries)
    allow_sensitive = canonical_overrides.get("allow_sensitive", args.allow_sensitive)
    execute_flag = canonical_overrides.get("execute", args.execute)
    autonomy_budget = canonical_overrides.get("autonomy_budget")
    if args.autonomy_budget:
        autonomy_budget = json.loads(args.autonomy_budget)
    else:
        autonomy_budget = dict(autonomy_budget or {})
        if args.budget_max_steps is not None:
            autonomy_budget["max_steps"] = int(args.budget_max_steps)
        if args.budget_max_retries is not None:
            autonomy_budget["max_retries"] = int(args.budget_max_retries)
        if args.budget_max_duration is not None:
            autonomy_budget["max_duration_sec"] = float(args.budget_max_duration)
        if args.budget_max_risk is not None:
            autonomy_budget["max_risk_score"] = float(args.budget_max_risk)
        if not autonomy_budget:
            autonomy_budget = None
    safety_contract = canonical_overrides.get("safety_contract")
    if args.safety_contract:
        safety_contract = json.loads(args.safety_contract)
    ask_on_uncertainty = bool(canonical_overrides.get("ask_on_uncertainty", False) or args.ask_on_uncertainty)
    spec = MissionSpec(
        instruction=instruction,
        constraints=constraints,
        timeout_secs=int(timeout),
        max_retries=int(max_retries),
        allow_sensitive=bool(allow_sensitive),
        execute=bool(execute_flag),
        autonomy_budget=autonomy_budget,
        safety_contract=safety_contract,
        ask_on_uncertainty=ask_on_uncertainty,
    )
    artifacts_dir = Path(args.artifacts_dir)
    executor = _resolve_executor(
        artifacts_dir,
        debug_browser=args.debug_browser,
        demo=args.demo,
    )
    if args.resume:
        print("[RESUME] Restoring mission from checkpoint")
    result = run_mission_sync(spec, executor=executor, resume_from=args.resume)
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0 if result.status == "complete" else 1


def _resolve_executor(
    artifacts_dir: Path,
    *,
    debug_browser: bool = False,
    demo: bool = False,
) -> MissionExecutor:
    hook = os.environ.get(_TEST_EXECUTOR_ENV)
    if hook:
        module_name, _, attr = hook.partition(":")
        module = importlib.import_module(module_name)
        factory: Callable[[Path], Any] = getattr(module, attr)
        executor = factory(artifacts_dir)
        if hasattr(executor, "debug_browser"):
            setattr(executor, "debug_browser", debug_browser)
        _apply_demo_config(executor, artifacts_dir, demo)
        return executor
    settings = _build_executor_settings(artifacts_dir, demo)
    return MissionExecutor(
        settings=settings,
        artifacts_root=artifacts_dir,
        debug_browser=debug_browser,
    )


def _build_executor_settings(artifacts_dir: Path, demo: bool) -> Dict[str, Any]:
    settings = load_settings()
    logging_cfg = settings.setdefault("logging", {})
    logging_cfg["artifact_root"] = str(artifacts_dir)
    settings["demo"] = demo
    return settings


def _apply_demo_config(executor: Any, artifacts_dir: Path, demo: bool) -> None:
    settings = getattr(executor, "settings", None)
    if isinstance(settings, dict):
        logging_cfg = settings.setdefault("logging", {})
        logging_cfg["artifact_root"] = str(artifacts_dir)
        settings["demo"] = demo
    elif demo:
        setattr(executor, "demo_mode", True)


if __name__ == "__main__":  # pragma: no cover - exercised via CLI test
    sys.exit(main())


def _load_canonical_spec(slug: str, manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Canonical mission manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    missions = payload.get("missions") if isinstance(payload, dict) else None
    if not isinstance(missions, list):
        raise ValueError(f"Invalid canonical mission manifest: {manifest_path}")
    for entry in missions:
        if isinstance(entry, dict) and entry.get("slug") == slug:
            return entry
    raise ValueError(f"Canonical mission '{slug}' not found in {manifest_path}")
