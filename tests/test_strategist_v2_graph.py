import asyncio
from typing import Any, Dict, List, Tuple

import pytest

from src.strategist.strategist_v1 import Strategist


def _make_worker(label: str, log: List[Tuple[str, List[str]]]):
    class _Worker:
        async def run(self, description: str, prev_results: Dict[str, Any]) -> Dict[str, Any]:
            log.append((label, list(prev_results.keys())))
            await asyncio.sleep(0)
            return {
                "label": label,
                "description": description,
                "seen": list(prev_results.keys()),
            }

    return _Worker


@pytest.mark.asyncio
async def test_strategist_builds_and_executes_dag() -> None:
    execution_log: List[Tuple[str, List[str]]] = []
    worker_registry = {
        "WorkerA": _make_worker("WorkerA", execution_log),
        "WorkerB": _make_worker("WorkerB", execution_log),
        "WorkerC": _make_worker("WorkerC", execution_log),
    }
    strategist = Strategist(worker_registry=worker_registry)

    plan = {
        "nodes": [
            {"id": "task1", "worker": "WorkerA", "desc": "analyze"},
            {"id": "task2", "worker": "WorkerB", "desc": "collect"},
            {"id": "task3", "worker": "WorkerC", "desc": "summarize"},
        ],
        "edges": [["task1", "task3"], ["task2", "task3"]],
    }

    dag = strategist.build_dag(plan)

    levels = dag.topological_levels()
    assert len(levels) == 2
    assert {node.node_id for node in levels[0]} == {"task1", "task2"}
    assert [node.node_id for node in levels[1]] == ["task3"]

    results = await strategist.execute_dag(dag, worker_registry)

    assert set(results.keys()) == {"task1", "task2", "task3"}
    first_level = [entry for entry in execution_log if entry[0] in {"WorkerA", "WorkerB"}]
    assert all(prev == [] for _, prev in first_level)

    final_entry = [entry for entry in execution_log if entry[0] == "WorkerC"][0]
    assert set(final_entry[1]) == {"task1", "task2"}