from __future__ import annotations

from eikon_engine.planning.planner_v3 import plan_from_goal


def test_score_penalizes_extra_steps() -> None:
    simple_plan = plan_from_goal("Open https://example.com", context={})
    complex_plan = plan_from_goal("Open https://example.com and then log in", context={})

    assert simple_plan["meta"]["score"] > complex_plan["meta"]["score"]
