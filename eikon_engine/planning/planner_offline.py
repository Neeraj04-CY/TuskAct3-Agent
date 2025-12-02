"""Deterministic planner used in tests and offline demos."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from eikon_engine.core.completion import build_completion
from eikon_engine.core.types import BrowserAction
from .memory_store import MemoryStore


_DEMO_SITE_PATH = Path(__file__).resolve().parents[2] / "examples" / "demo_local_testsite" / "login.html"
_DEMO_SITE_URL = _DEMO_SITE_PATH.as_uri() if _DEMO_SITE_PATH.exists() else "https://the-internet.herokuapp.com/login"


HEROKU_DEFAULT_STEPS: List[BrowserAction] = [
    {"action": "navigate", "url": _DEMO_SITE_URL},
    {"action": "fill", "selector": "#username", "value": "tomsmith"},
    {"action": "fill", "selector": "#password", "value": "SuperSecretPassword!"},
    {"action": "click", "selector": "button[type=\"submit\"]"},
    {"action": "screenshot", "name": "heroku_login.png"},
    {"action": "extract_dom"},
]


class OfflinePlanner:
    """Rule-based planner that emits deterministic actions."""

    def __init__(self, *, memory_store: MemoryStore | None = None) -> None:
        self.memory_store = memory_store or MemoryStore()

    async def create_plan(self, goal: str) -> Dict[str, Any]:
        """Return a richer action list tailored to the provided goal."""

        tokens = goal.lower()
        actions: List[BrowserAction]
        if "heroku" in tokens or "secure area" in tokens:
            actions = HEROKU_DEFAULT_STEPS.copy()
        else:
            actions = [
                {"action": "navigate", "url": "https://example.com"},
                {"action": "screenshot", "name": "page.png"},
                {"action": "extract_dom"},
            ]

        related = self.memory_store.retrieve(goal, limit=2)
        if any("logout" in record.text.lower() for record in related):
            actions.append({"action": "click", "selector": "a.logout"})
        if all(step.get("action") != "extract_dom" for step in actions):
            actions.append({"action": "extract_dom"})

        self.memory_store.add(goal[:40], goal)
        self.memory_store.remember({"summary": f"plan:{goal}", "actions": len(actions)})
        completion = build_completion(
            complete=True,
            reason="plan generated",
            payload={"steps": len(actions), "memory_hits": len(related)},
        )
        return {
            "goal": goal,
            "actions": actions,
            "related_memories": [record.text for record in related],
            "completion": completion,
        }
