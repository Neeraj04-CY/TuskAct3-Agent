"""Minimal example showing how to run the browser pipeline."""

from __future__ import annotations

import asyncio
import json

from eikon_engine.pipelines.browser_pipeline import run_browser_goal


async def _main() -> None:
    goal = "Log in to the Heroku demo site and confirm the secure area message."
    result = await run_browser_goal(goal)
    print(json.dumps(result, indent=2))


def run() -> None:
    """Entry point used by documentation and scripts."""

    asyncio.run(_main())


if __name__ == "__main__":
    run()
