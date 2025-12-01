from __future__ import annotations

import numpy as np
import pytest

from src.autonomous.autonomous_runner import AutonomousRunner
from src.memory.memory_manager import MemoryManager
from src.strategist.strategist_v1 import Strategist


class AnalyzerWorker:
    async def run(self, description: str, prev_results: dict) -> dict:
        return {"analysis": "summary available", "description": description}


class GeneratorWorker:
    async def run(self, description: str, prev_results: dict) -> dict:
        return {"generation": "done", "description": description}


@pytest.mark.asyncio
async def test_autonomous_runner_goal_chaining() -> None:
    memory_manager = MemoryManager()
    memory_manager._store.set_embedder(lambda text: np.ones(8, dtype="float32"))

    worker_registry = {
        "WorkerA": AnalyzerWorker,
        "WorkerB": GeneratorWorker,
        "WorkerC": GeneratorWorker,
        "WorkerD": GeneratorWorker,
        "reflection_worker": GeneratorWorker,
    }

    strategist = Strategist(worker_registry=worker_registry, memory_manager=memory_manager)
    runner = AutonomousRunner(strategist=strategist, max_iters=3)

    payload = await runner.run("Research cats")

    assert payload["iterations"] >= 1
    assert payload["goals_run"]
    assert payload["trace"]
    assert payload["final_goal"]
    assert all("reflection" in step for step in payload["trace"])
