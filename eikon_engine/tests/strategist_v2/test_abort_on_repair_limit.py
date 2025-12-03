from __future__ import annotations

from eikon_engine.strategist.strategist_v2 import StrategistV2


class StubPlanner:
    async def create_plan(self, goal: str, last_result=None):  # noqa: D401
        return {"plan_id": "stub", "goal": goal, "tasks": []}


def make_strategist(**kwargs) -> StrategistV2:
    return StrategistV2(planner=StubPlanner(), **kwargs)


def test_abort_after_repair_budget_exhausted() -> None:
    strategist = make_strategist(failure_budget=2)
    strategist._repair_attempts = strategist.failure_budget  # type: ignore[attr-defined]
    assert strategist.should_abort()


def test_abort_after_repeated_failure_type() -> None:
    strategist = make_strategist(failure_limit=2)
    strategist.record_failure("timeout")
    strategist.record_failure("timeout")
    assert strategist.should_abort()
