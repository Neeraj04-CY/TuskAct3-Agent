import asyncio
from typing import Any, Dict

import pytest

from eikon_engine.workflow.workflow_engine import execute_graph
from src.strategist.strategist_v1 import Strategist


def _worker_factory(label: str):
    class _Worker:
        async def run(self, description: str, prev_results: Dict[str, Any]) -> Dict[str, Any]:
            await asyncio.sleep(0)
            return {"label": label, "description": description, "seen": list(prev_results.keys())}

    return _Worker


@pytest.mark.asyncio
async def test_workflow_engine_matches_strategist_execution() -> None:
    registry = {
        "WorkerA": _worker_factory("WorkerA"),
        "WorkerB": _worker_factory("WorkerB"),
        "WorkerC": _worker_factory("WorkerC"),
    }

    strategist = Strategist(worker_registry=registry)
    plan = {
        "nodes": [
            {"id": "task1", "worker": "WorkerA", "desc": "prep"},
            {"id": "task2", "worker": "WorkerB", "desc": "draft"},
            {"id": "task3", "worker": "WorkerC", "desc": "finalize"},
        ],
        "edges": [["task1", "task3"], ["task2", "task3"]],
    }

    dag = strategist.build_dag(plan)

    strategist_results = await strategist.execute_dag(dag, registry)
    workflow_results = await execute_graph(dag, registry)

    assert workflow_results == strategist_results
