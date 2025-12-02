"""Autonomous multi-goal browser demo using the WebAgent API."""

from __future__ import annotations

import asyncio
import json

from eikon_engine.core.goal_manager import GoalManager
from eikon_engine.core.web_agent import WebAgent

INSTRUCTION = """
Open the Heroku sample login page, sign in with the demo credentials, capture the secure area
message, and log out.
""".strip()


async def _main() -> None:
    manager = GoalManager.parse(INSTRUCTION)
    async with WebAgent() as agent:
        result = await agent.run_with_goals(manager)
    print(json.dumps(result, indent=2))


def run() -> None:
    """Entry point compatible with python -m eikon_engine.examples.autonomous_demo."""

    asyncio.run(_main())


if __name__ == "__main__":  # pragma: no cover
    run()
