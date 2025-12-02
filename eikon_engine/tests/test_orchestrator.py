from __future__ import annotations

import pytest

from eikon_engine.core.completion import build_completion
from eikon_engine.core.orchestrator import build_orchestrator
from eikon_engine.core.strategist import Strategist
from eikon_engine.planning.planner_offline import OfflinePlanner


class DummyWorker:
    def __init__(self) -> None:
        self.invocations = 0

    async def execute(self, metadata):
        self.invocations += 1
        return {
            "steps": [metadata],
            "screenshots": [],
            "dom_snapshot": "<html></html>",
            "layout_graph": "html>body",
            "completion": build_completion(complete=self.invocations >= 2, reason="ok"),
            "error": None,
        }


@pytest.mark.asyncio
async def test_orchestrator_respects_completion():
    planner = OfflinePlanner()
    strategist = Strategist(planner=planner)
    worker = DummyWorker()

    orchestrator = build_orchestrator(strategist=strategist, worker=worker, settings={"completion": {"max_steps": 5}})
    result = await orchestrator.run("Visit example.com")

    assert result["completion"]["complete"] is True
    assert worker.invocations == 2
    assert len(result["steps"]) == 2
