import pytest

from src.strategist.strategist_v1 import Strategist


class AnalyzerWorker:
    async def run(self, description: str, prev_results: dict):
        return {"analysis": "ok", "description": description}


class GeneratorWorker:
    async def run(self, description: str, prev_results: dict):
        return {"generation": "ok", "description": description}


@pytest.mark.asyncio
async def test_strategist_smoke():
    worker_registry = {
        "analyzer": AnalyzerWorker,
        "generator": GeneratorWorker,
    }
    strategist = Strategist(worker_registry=worker_registry)

    result = await strategist.run("Write a short report")

    assert "parsed_goal" in result
    assert "plan" in result
    assert "results" in result
    assert result["memory_written"] is None

    outputs = result["results"].values()
    assert any(output.get("analysis") == "ok" for output in outputs)
    assert any(output.get("generation") == "ok" for output in outputs)
