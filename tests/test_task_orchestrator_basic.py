from __future__ import annotations

import pytest

from src.task_orchestrator import TaskOrchestrator
from tests.phase4_fakes import FakeStrategist, StubMemoryManager


class SimpleWorker:
    async def run(self, description: str, prev_results: dict) -> dict:
        return {"description": description, "prev": list(prev_results.keys())}


class ImmediateReflection:
    async def run(self, description: str, prev_results: dict) -> dict:
        return {
            "goal_satisfied": True,
            "next_goal": None,
            "intent_corrected": False,
            "repair_suggestion": None,
            "notes": "task complete",
        }


@pytest.mark.asyncio
async def test_task_orchestrator_basic_transcript() -> None:
    memory = StubMemoryManager()
    worker_registry = {"WorkerA": SimpleWorker}
    plan_nodes = [
        {"id": "task1", "worker": "WorkerA", "desc": "Collect info"},
    ]
    strategist = FakeStrategist(worker_registry, plan_nodes, edges=[], memory_manager=memory)

    orchestrator = TaskOrchestrator(
        strategist=strategist,
        memory_manager=memory,
        reflection_worker=ImmediateReflection(),
    )

    payload = await orchestrator.execute("Basic goal")

    assert payload["status"] == "success"
    assert len(payload["transcript"]) == 1
    entry = payload["transcript"][0]
    assert entry["result"]["task1"]["description"] == "Collect info"
    assert entry["reflection"]["goal_satisfied"] is True
    assert entry["memory_written"] is not None
    assert payload["summary"] == "task complete"
