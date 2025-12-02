from __future__ import annotations

from eikon_engine.planning.planner_v3 import plan_from_goal


def test_pipeline_order_sequence() -> None:
    plan = plan_from_goal("Open https://example.com", context={})
    assert plan["meta"]["pipeline_order"] == [
        "raw",
        "grouping",
        "durability",
        "prechecks",
        "recovery",
        "scoring",
    ]
