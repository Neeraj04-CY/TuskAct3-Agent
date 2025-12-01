from typing import Any, Dict

import pytest

from src.strategist.strategist_v1 import Strategist


class DummyAnalyzer:
    async def run(self, description: str, prev_results: Dict[str, Any]) -> Dict[str, Any]:
        return {"analysis": "done", "received": list(prev_results.keys())}


class DummyGenerator:
    async def run(self, description: str, prev_results: Dict[str, Any]) -> Dict[str, Any]:
        return {"generation": "done", "received": list(prev_results.keys())}


@pytest.mark.asyncio
async def test_strategist_v1_end_to_end() -> None:
    worker_registry = {
        "analyzer": DummyAnalyzer,
        "generator": DummyGenerator,
    }
    strategist = Strategist(worker_registry=worker_registry)

    result = await strategist.run("Test goal")

    assert {"parsed_goal", "plan", "results"}.issubset(result)
    assert result["parsed_goal"]["raw"] == "Test goal"
    assert result["memory_written"] is None
    assert "memory_write_error" not in result

    plan = result["plan"]
    assert "nodes" in plan and "edges" in plan
    assert len(plan["nodes"]) >= 2

    outputs = list(result["results"].values())
    assert any("analysis" in output for output in outputs)
    assert any("generation" in output for output in outputs)