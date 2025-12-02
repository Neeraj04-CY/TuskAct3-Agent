"""Autonomous multi-goal browser demo."""

from __future__ import annotations

import json

from eikon_engine.api import WebAgent

INSTRUCTION = "Go to HerokuApp, log in with demo creds, capture the secure header text, and log out."


def main() -> None:
    agent = WebAgent()
    result = agent.run(INSTRUCTION)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
