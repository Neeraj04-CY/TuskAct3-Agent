from __future__ import annotations

import asyncio

from src.memory.memory_manager import MemoryManager
from src.strategist.workflow_builder import Strategist
from src.worker.executor import Worker
from src.workflow.engine import WorkflowEngine, WorkflowEngineConfig


def test_workflow_engine_end_to_end() -> None:
    strategist = Strategist()
    worker = Worker()
    memory_manager = MemoryManager()
    engine = WorkflowEngine(
        strategist=strategist,
        worker=worker,
        memory_manager=memory_manager,
        config=WorkflowEngineConfig(enable_memory=False),
    )

    result = asyncio.run(engine.run("Test simple workflow"))
    assert "workflow_id" in result
    assert result["task"] == "Test simple workflow"
    assert len(result["results"]) >= 1