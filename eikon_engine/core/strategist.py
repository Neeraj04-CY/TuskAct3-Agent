"""Strategist responsible for translating goals into browser steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol

from eikon_engine.core.completion import build_completion
from eikon_engine.core.types import AnyDict, BrowserAction, CompletionPayload
from eikon_engine.planning.memory_store import MemoryStore


@dataclass
class StrategyStep:
    """Represents a single step returned by the Strategist."""

    description: str
    metadata: Dict[str, Any]


class Strategist:
    """Simple strategist that feeds steps to the orchestrator."""

    def __init__(self, *, planner: "PlannerProtocol", memory_store: MemoryStore | None = None) -> None:
        self._planner = planner
        self.memory_store = memory_store or getattr(planner, "memory_store", MemoryStore())
        self._pending: List[StrategyStep] = []
        self._completed: List[AnyDict] = []
        self._last_completion: CompletionPayload | None = None
        self._goal: str = ""
        self._last_result: Optional[Dict[str, Any]] = None

    async def initialize(self, goal: str) -> None:
        """Populate the queue with initial steps derived from the planner."""

        self._goal = goal
        self._pending.clear()
        self._completed.clear()
        self._last_completion = None
        self._last_result = None
        await self._load_plan(goal)

    def has_next(self) -> bool:
        """Return True if there are pending steps."""

        return bool(self._pending)

    def next_step(self) -> StrategyStep:
        """Pop the next step from the queue."""

        if not self._pending:
            raise StopIteration("No more strategy steps available")
        return self._pending.pop(0)

    async def record_result(self, result: Dict[str, Any]) -> None:
        """Record a worker result and update memory/completion state."""

        self._completed.append(result)
        self._last_result = result
        summary = result.get("completion", {}).get("reason") or result.get("error") or "step"
        self.memory_store.remember({"summary": summary, "result": result})
        completion = result.get("completion")
        if isinstance(completion, dict) and completion.get("complete"):
            self._last_completion = completion
            self._pending.clear()
            return

    def completion_state(self) -> CompletionPayload:
        """Return the last seen completion payload or a default incomplete one."""

        if self._last_completion:
            return self._last_completion
        return build_completion(complete=False, reason="strategy pending", payload={})

    async def ensure_plan(self) -> None:
        """Replenish pending steps using the most recent feedback if the queue is empty."""

        if self._pending or not self._goal:
            return
        await self._load_plan(self._goal, last_result=self._last_result)

    async def _load_plan(self, goal: str, last_result: Optional[Dict[str, Any]] = None) -> None:
        plan = await self._invoke_planner(goal, last_result=last_result)
        actions: Iterable[BrowserAction] = plan.get("actions", [])
        self._pending = [
            StrategyStep(description=str(action), metadata={"action": action})
            for action in actions
        ]

    async def _invoke_planner(
        self,
        goal: str,
        *,
        last_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            return await self._planner.create_plan(goal, last_result=last_result)
        except TypeError:
            return await self._planner.create_plan(goal)


class PlannerProtocol(Protocol):
    """Sub-set of planner capabilities consumed by Strategist."""

    async def create_plan(self, goal: str, *, last_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return a planner response with actions."""

