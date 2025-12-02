from __future__ import annotations

import pytest

from eikon_engine.core.completion import build_completion, is_complete
from eikon_engine.core.strategist import Strategist
from eikon_engine.planning.planner_offline import OfflinePlanner


@pytest.mark.asyncio
async def test_offline_planner_completion_contract():
    planner = OfflinePlanner()
    goal = "log into the sample heroku app"
    plan = await planner.create_plan(goal)
    assert plan["completion"]["complete"] is True
    assert plan["completion"]["payload"]["steps"] == len(plan["actions"])

    strategist = Strategist(planner=planner)
    await strategist.initialize(goal)
    assert strategist.has_next() is True
    await strategist.record_result({"completion": plan["completion"]})
    assert strategist.completion_state()["complete"] is True


def test_is_complete_guard():
    result = {"completion": build_completion(complete=True, reason="done")}
    assert is_complete(result) is True
    assert is_complete({}) is False
