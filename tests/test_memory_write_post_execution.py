from __future__ import annotations

import numpy as np
import pytest

from src.memory.memory_manager import MemoryManager
from src.strategist.strategist_v1 import Strategist


class SimpleWorker:
    async def run(self, description: str, prev_results: dict) -> dict:
        return {"description": description, "seen": list(prev_results.keys())}


@pytest.mark.asyncio
async def test_strategist_writes_memory_post_execution() -> None:
    memory_manager = MemoryManager()
    memory_manager._store.set_embedder(lambda text: np.ones(8, dtype="float32"))

    strategist = Strategist(
        worker_registry={"worker": SimpleWorker},
        memory_manager=memory_manager,
    )

    result = await strategist.run("Document new workflow behavior")

    assert result["memory_written"] is not None
    memory_id = result["memory_written"]
    assert memory_id in memory_manager._store._metadata
    stored_metadata = memory_manager._store._metadata[memory_id]
    assert stored_metadata["context"]["worker_count"] >= 1
    assert "memory_write_error" not in result
