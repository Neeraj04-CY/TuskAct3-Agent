from __future__ import annotations

import asyncio

import pytest

from eikon_engine.core.orchestrator_v2 import OrchestratorV2
from eikon_engine.strategist.strategist_v2 import StrategistV2


class StubPlanner:
    async def create_plan(self, goal: str, last_result=None):  # noqa: D401
        return {
            "plan_id": "demo",
            "goal": goal,
            "tasks": [
                {
                    "id": "task_1",
                    "tool": "BrowserWorker",
                    "bucket": "nav",
                    "depends_on": [],
                    "inputs": {
                        "actions": [
                            {"id": "s1", "action": "navigate", "url": "https://site"},
                        ]
                    },
                }
            ],
        }


class StubWorker:
    async def execute(self, metadata):  # noqa: D401
        return {
            "completion": {"complete": True, "reason": "navigate"},
            "dom_snapshot": "<div>dashboard</div>",
            "meta": {"url": "https://site/dashboard"},
            "steps": [],
        }

    async def close(self) -> None:  # noqa: D401
        return None


@pytest.mark.asyncio
async def test_reward_trace_is_emitted() -> None:
    strategist = StrategistV2(planner=StubPlanner())
    worker = StubWorker()
    orchestrator = OrchestratorV2(strategist=strategist, worker=worker)
    result = await orchestrator.run_goal("demo")
    reward_trace = result["run_context"].get("reward_trace")
    assert reward_trace
    assert "confidence" in reward_trace[0]
    assert reward_trace[0]["confidence"]["band"] in {"low", "medium", "high"}
