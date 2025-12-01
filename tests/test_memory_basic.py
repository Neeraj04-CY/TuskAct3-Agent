from __future__ import annotations

import pytest

from src.memory.memory_manager import MemoryManager
from src.strategist.strategist_v1 import Strategist


class EchoWorker:
    async def run(self, description: str, prev_results: dict) -> dict:
        return {"description": description, "seen": list(prev_results.keys())}


@pytest.mark.asyncio
async def test_memory_manager_add_and_query_async() -> None:
    memory = MemoryManager()

    await memory.add_memory(
        event_type="observation",
        title="UnitTest",
        text="Remember to validate hashed embeddings",
        context={"source": "unit"},
    )

    matches = await memory.query_similar("Remember to validate hashed embeddings", k=1)

    assert matches
    assert matches[0]["metadata"]["title"] == "UnitTest"


@pytest.mark.asyncio
async def test_strategist_injects_related_memories() -> None:
    memory = MemoryManager()
    await memory.add_memory(
        event_type="analysis",
        title="Solar",
        text="Analyze solar flare telemetry",
        context={"scope": "astrophysics"},
    )

    strategist = Strategist(worker_registry={"WorkerA": EchoWorker}, memory_manager=memory)

    result = await strategist.run("Analyze solar flare telemetry for anomalies")

    related = result["parsed_goal"].get("related_memories")
    assert related is not None
    assert len(related) >= 1
    assert related[0]["metadata"]["title"] == "Solar"
