from __future__ import annotations

import asyncio

from src.task_orchestrator import TaskOrchestrator
from src.strategist.strategist_v1 import Strategist
from src.workers.high_level.reflection_worker import ReflectionWorker


class _CompleteWorker:
    async def run(self, description: str, prev_results: dict[str, str], **_: str) -> dict[str, object]:
        return {
            "echo": description,
            "completion": {"success": True, "complete": True, "reason": "Finished"},
        }


def _registry() -> dict[str, type[_CompleteWorker]]:
    return {name: _CompleteWorker for name in ["WorkerA", "WorkerB", "WorkerC", "WorkerD", "browser"]}


def test_completion_propagation() -> None:
    strategist = Strategist(worker_registry=_registry())
    result = asyncio.run(strategist.run("Complete the task"))
    assert result["completion"]["complete"] is True
    assert result["completion"]["reason"] == "Finished"


def test_orchestrator_stops_on_completion() -> None:
    strategist = Strategist(worker_registry=_registry())
    orchestrator = TaskOrchestrator(strategist=strategist)
    outcome = asyncio.run(orchestrator.execute("Complete the task", max_iters=3))
    assert outcome["status"] == "completed"
    assert outcome["completion"]["complete"] is True
    assert len(outcome["transcript"]) == 1


def test_reflection_stops_on_complete_flag() -> None:
    reflection_worker = ReflectionWorker()
    payload = {
        "plan": {},
        "results": {},
        "completion": {"success": True, "complete": True, "reason": "Finished"},
    }
    result = asyncio.run(reflection_worker.run("Complete the task", payload))
    assert result["goal_satisfied"] is True
    assert result["next_goal"] is None
    assert "Finished" in result["notes"]
