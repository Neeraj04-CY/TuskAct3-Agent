from __future__ import annotations

from typing import Any, Dict, List

from src.memory.memory_manager import MemoryManager, MemoryRecord


class MemoryIndexer:
    """
    Helper responsible for turning raw logs / outcomes into MemoryRecord objects.

    v1: expects embedding already available (no embedding model wired yet).
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        self._memory = memory_manager

    def index_failure(self, workflow_id: str, step_id: str, error: str, embedding: List[float]) -> None:
        record = MemoryRecord(
            kind="failure",
            payload={
                "workflow_id": workflow_id,
                "step_id": step_id,
                "error": error
            },
            embedding=embedding,
        )
        self._memory.store("failures", record)

    def index_success(self, workflow_id: str, output: Dict[str, Any], embedding: List[float]) -> None:
        record = MemoryRecord(
            kind="success",
            payload={
                "workflow_id": workflow_id,
                "output": output
            },
            embedding=embedding,
        )
        self._memory.store("successes", record)