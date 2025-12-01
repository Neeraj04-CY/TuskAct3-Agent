from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Dict

from src.memory.memory_manager import MemoryManager
from src.strategist.strategist_v1 import Strategist
from src.task_orchestrator import TaskOrchestrator
from src.workers.browser import BrowserWorker


async def run_direct_worker(actions: list[Dict[str, str]], *, allow_sensitive: bool) -> None:
    worker = BrowserWorker()
    payload = await worker.run(json.dumps(actions), prev_results={}, dry_run=True, allow_sensitive=allow_sensitive)
    print("Dry-run result:")
    print(json.dumps(payload, indent=2))

    if os.getenv("PLAYWRIGHT_BYPASS_DRY_RUN", "0") == "1":
        print("\nPLAYWRIGHT_BYPASS_DRY_RUN set – executing real browser session...")
        live_payload = await worker.run(
            json.dumps(actions),
            prev_results={},
            dry_run=False,
            allow_sensitive=allow_sensitive,
        )
        print(json.dumps(live_payload, indent=2))
    else:
        print("\nSet PLAYWRIGHT_BYPASS_DRY_RUN=1 to execute the live run.")


async def run_strategist_demo(goal: str) -> None:
    worker_registry = {
        "browser": BrowserWorker,
        "WorkerA": BrowserWorker,
        "WorkerB": BrowserWorker,
        "WorkerC": BrowserWorker,
        "WorkerD": BrowserWorker,
    }
    strategist = Strategist(worker_registry=worker_registry, memory_manager=MemoryManager())
    orchestrator = TaskOrchestrator(strategist=strategist)

    payload = await orchestrator.execute(goal, max_iters=1)
    print("\nTaskOrchestrator transcript:")
    print(json.dumps(payload, indent=2))
    completion = payload.get("completion")
    if completion and completion.get("complete"):
        print(f"Workflow complete: {completion.get('reason', 'success')}")
    else:
        print(f"Workflow status: {payload.get('status')}")
    if os.getenv("PLAYWRIGHT_BYPASS_DRY_RUN", "0") != "1":
        print("BrowserWorker stayed in dry-run mode. Set PLAYWRIGHT_BYPASS_DRY_RUN=1 for live automation.")


async def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    demo_page = repo_root / "examples" / "demo_local_testsite" / "login.html"
    demo_url = demo_page.resolve().as_uri()
    allow_sensitive = os.getenv("EIKON_ALLOW_SENSITIVE", "0") == "1"

    actions = [
        {"action": "navigate", "url": demo_url},
        {"action": "fill", "selector": "#username", "value": "tomsmith"},
        {"action": "fill", "selector": "#password", "value": "SuperSecretPassword!"},
        {"action": "click", "selector": "#login"},
        {"action": "screenshot", "name": "demo_login.png"},
        {"action": "extract_dom"},
    ]

    await run_direct_worker(actions, allow_sensitive=allow_sensitive)

    goal = (
        f"Navigate to {demo_url} and log in. "
        "Fill #username with tomsmith. Fill #password with SuperSecretPassword!. Click the Login button."
    )
    await run_strategist_demo(goal)


if __name__ == "__main__":
    asyncio.run(main())
