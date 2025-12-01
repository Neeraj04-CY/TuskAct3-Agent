from __future__ import annotations

import pytest

from src.task_orchestrator import TaskOrchestrator
from tests.phase4_fakes import FakeStrategist, StubMemoryManager


class ParallelWorker:
    def __init__(self) -> None:
        self.invocations: list[str] = []

    async def run(self, description: str, prev_results: dict) -> dict:
        self.invocations.append(description)
        return {"worker": description}


class TwoPhaseReflection:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, description: str, prev_results: dict) -> dict:
        self.calls += 1
        if self.calls == 1:
            return {
                "goal_satisfied": False,
                "next_goal": "Follow-up",
                "intent_corrected": False,
                "repair_suggestion": None,
                "notes": "retry",
            }
        return {
            "goal_satisfied": True,
            "next_goal": None,
            "intent_corrected": False,
            "repair_suggestion": None,
            "notes": "done",
        }


@pytest.mark.asyncio
async def test_task_orchestrator_runs_parallel_steps() -> None:
    memory = StubMemoryManager()
    worker_registry = {"WorkerA": ParallelWorker, "WorkerB": ParallelWorker}
    plan_nodes = [
        {"id": "task1", "worker": "WorkerA", "desc": "Parallel task 1"},
        {"id": "task2", "worker": "WorkerB", "desc": "Parallel task 2"},
    ]
    strategist = FakeStrategist(worker_registry, plan_nodes, edges=[], memory_manager=memory)

    orchestrator = TaskOrchestrator(
        strategist=strategist,
        memory_manager=memory,
        reflection_worker=TwoPhaseReflection(),
    )

    payload = await orchestrator.execute("Parallel goal")

    assert payload["status"] == "success"
    assert len(payload["transcript"]) == 2
    first_entry = payload["transcript"][0]
    assert set(first_entry["result"].keys()) == {"task1", "task2"}
    assert payload["summary"].splitlines()[-1] == "done"
