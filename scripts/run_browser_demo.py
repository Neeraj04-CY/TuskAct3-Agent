from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eikon_engine.browser.worker_v1 import BrowserWorkerV1
from eikon_engine.pipelines.browser_pipeline import run_browser_goal


def _build_plan(actions: list[Dict[str, Any]], goal: str) -> Dict[str, Any]:
    return {
        "plan_id": "demo-plan",
        "goal": goal,
        "tasks": [
            {
                "id": "task_1",
                "tool": "BrowserWorker",
                "inputs": {"actions": actions},
                "depends_on": [],
                "bucket": "demo",
            }
        ],
    }


async def _execute_manual_worker(
    actions: list[Dict[str, Any]],
    *,
    goal: str,
    enable_playwright: bool,
    allow_external: bool,
) -> Dict[str, Any]:
    worker = BrowserWorkerV1(
        settings={"browser": {"allow_external": allow_external}},
        enable_playwright=enable_playwright,
    )
    plan = _build_plan(actions, goal)
    summary = await worker.run_plan(plan, goal=goal)
    await worker.close()
    return summary


async def run_direct_worker(actions: list[Dict[str, Any]], *, allow_external: bool) -> None:
    goal = "Demo plan"
    dry_summary = await _execute_manual_worker(
        actions,
        goal=goal,
        enable_playwright=False,
        allow_external=allow_external,
    )
    print("Dry-run summary (no Playwright):")
    print(json.dumps(dry_summary, indent=2))

    if os.getenv("PLAYWRIGHT_BYPASS_DRY_RUN", "0") == "1":
        print("\nPLAYWRIGHT_BYPASS_DRY_RUN set – executing with Playwright...")
        live_summary = await _execute_manual_worker(
            actions,
            goal=goal,
            enable_playwright=True,
            allow_external=allow_external,
        )
        print(json.dumps(live_summary, indent=2))
    else:
        print("\nSet PLAYWRIGHT_BYPASS_DRY_RUN=1 to launch a real browser session.")


async def run_strategist_demo(goal: str) -> None:
    result = await run_browser_goal(goal)
    print("\nPlanner → Strategist → Orchestrator transcript:")
    print(json.dumps(result, indent=2))
    completion = result.get("completion") or {}
    if completion.get("complete"):
        print(f"Workflow complete: {completion.get('reason', 'success')}")
    else:
        print("Workflow incomplete; inspect transcript for details.")
        print("BrowserWorker stayed in dry-run mode. Set PLAYWRIGHT_BYPASS_DRY_RUN=1 for live automation.")


async def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    demo_page = repo_root / "examples" / "demo_local_testsite" / "login.html"
    demo_url = demo_page.resolve().as_uri()
    allow_external_env = os.getenv("EIKON_ALLOW_EXTERNAL")
    if allow_external_env is None:
        allow_external_env = os.getenv("EIKON_ALLOW_SENSITIVE", "0")
    allow_external = allow_external_env == "1"

    actions = [
        {"action": "navigate", "url": demo_url},
        {"action": "fill", "selector": "#username", "value": "tomsmith"},
        {"action": "fill", "selector": "#password", "value": "SuperSecretPassword!"},
        {"action": "click", "selector": "#login"},
        {"action": "screenshot", "name": "demo_login.png"},
        {"action": "extract_dom"},
    ]

    await run_direct_worker(actions, allow_external=allow_external)

    goal = (
        f"Navigate to {demo_url} and log in. "
        "Fill #username with tomsmith. Fill #password with SuperSecretPassword!. Click the Login button."
    )
    await run_strategist_demo(goal)


if __name__ == "__main__":
    asyncio.run(main())
