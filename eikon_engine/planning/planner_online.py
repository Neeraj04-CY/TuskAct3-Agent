"""Pseudo online planner that enriches the offline plan with memory."""

from __future__ import annotations

from typing import Any, Dict, List

from eikon_engine.core.completion import build_completion
from .memory_store import MemoryStore
from .planner_offline import OfflinePlanner


class OnlinePlanner:
    """Thin wrapper mimicking an online LLM-backed planner."""

    def __init__(self, *, memory_store: MemoryStore | None = None) -> None:
        self.memory_store = memory_store or MemoryStore()
        self._offline = OfflinePlanner(memory_store=self.memory_store)

    async def create_plan(self, goal: str, *, last_result: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Generate or repair a plan using feedback from previous steps."""

        plan = await self._offline.create_plan(goal)
        actions: List[Dict[str, Any]] = list(plan.get("actions", []))

        if last_result:
            self.memory_store.remember({"summary": f"feedback:{goal}", **last_result})
            if last_result.get("error"):
                actions = self._fallback_actions(goal)
            elif not last_result.get("completion", {}).get("complete"):
                actions = self._rewrite_incomplete(actions, last_result)

        related = [record.text for record in self.memory_store.retrieve(goal)]
        plan.update(
            {
                "related_memories": related,
                "actions": actions,
                "mode": "online",
                "completion": build_completion(
                    complete=True,
                    reason="online planner generated plan",
                    payload={"steps": len(actions), "feedback_used": bool(last_result)},
                ),
            }
        )
        return plan

    def _fallback_actions(self, goal: str) -> List[Dict[str, Any]]:
        return [
            {"action": "navigate", "url": "https://the-internet.herokuapp.com/login"},
            {"action": "extract_dom"},
            {"action": "screenshot", "name": "fallback.png"},
        ]

    def _rewrite_incomplete(self, actions: List[Dict[str, Any]], result: Dict[str, Any]) -> List[Dict[str, Any]]:
        executed = len(result.get("steps") or [])
        remaining = actions[executed:] if executed < len(actions) else []
        if not remaining:
            remaining = actions[-2:] if len(actions) >= 2 else actions
        rewritten = list(remaining)
        rewritten.append({"action": "extract_dom"})
        if result.get("error"):
            rewritten.insert(0, {"action": "navigate", "url": "https://example.com"})
        return rewritten
