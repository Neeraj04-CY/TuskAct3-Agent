from __future__ import annotations

import numpy as np
import pytest

from src.autonomous.autonomous_runner import AutonomousRunner
from src.memory.memory_manager import MemoryManager
from src.strategist.strategist_v1 import Strategist
from src.workers.high_level.reflection_worker import ReflectionWorker


def _build_registry(worker_cls):
    return {
        "WorkerA": worker_cls,
        "WorkerB": worker_cls,
        "WorkerC": worker_cls,
        "WorkerD": worker_cls,
    }


class SummaryWorker:
    async def run(self, description: str, prev_results: dict) -> dict:
        return {"status": "summary complete", "description": description}


class ErrorWorker:
    async def run(self, description: str, prev_results: dict) -> dict:
        return {"error": "tool failure", "description": description}


class EmptyWorker:
    async def run(self, description: str, prev_results: dict) -> dict:
        return {}


class LoopReflectionWorker(ReflectionWorker):
    async def run(self, description: str, prev_results: dict) -> dict:
        return {
            "goal_satisfied": False,
            "next_goal": "Loop forever",
            "intent_corrected": False,
            "repair_suggestion": None,
            "notes": "loop",
        }


@pytest.mark.asyncio
async def test_autonomous_v2_intent_correction_and_memory_enrichment() -> None:
    memory_manager = MemoryManager()
    memory_manager._store.set_embedder(lambda text: np.ones(8, dtype="float32"))
    await memory_manager.add_memory(
        event_type="note",
        title="Cat Facts",
        text="Cats have retractable claws",
        context={"topic": "cats"},
    )

    strategist = Strategist(worker_registry=_build_registry(SummaryWorker), memory_manager=memory_manager)
    runner = AutonomousRunner(strategist=strategist, max_iters=2)

    payload = await runner.run("Cats")

    reflection = payload["trace"][0]["reflection"]
    assert reflection["intent_corrected"] is True
    assert "Memory insights" in reflection["notes"]


@pytest.mark.asyncio
async def test_autonomous_v2_repairs_failed_execution() -> None:
    memory_manager = MemoryManager()
    memory_manager._store.set_embedder(lambda text: np.ones(8, dtype="float32"))

    strategist = Strategist(worker_registry=_build_registry(ErrorWorker), memory_manager=memory_manager)
    runner = AutonomousRunner(strategist=strategist, max_iters=3)

    payload = await runner.run("Investigate failure")

    trace_entry = payload["trace"][0]
    reflection = trace_entry["reflection"]
    assert reflection["repair_suggestion"] is not None
    assert trace_entry["self_healing"] is True
    assert reflection["next_goal"].startswith("Repair")


@pytest.mark.asyncio
async def test_autonomous_v2_stops_on_repeated_next_goal() -> None:
    memory_manager = MemoryManager()
    memory_manager._store.set_embedder(lambda text: np.ones(8, dtype="float32"))

    strategist = Strategist(worker_registry=_build_registry(EmptyWorker), memory_manager=memory_manager)
    runner = AutonomousRunner(
        strategist=strategist,
        max_iters=5,
        reflection_worker=LoopReflectionWorker(),
    )

    payload = await runner.run("Loop test")

    assert payload["iterations"] == 2
    assert payload["trace"][-1]["loop_detected"] is True
