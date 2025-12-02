from __future__ import annotations

from eikon_engine.planning.planner_v3 import plan_from_goal


def test_plan_meta_includes_execution_bridge_fields() -> None:
    goal = "Open https://example.com/login then log in"
    context = {"credentials": {"username": "demo", "password": "pass"}}

    plan = plan_from_goal(goal, context=context)
    meta = plan["meta"]

    assert meta.get("durability_summary") is not None
    assert meta.get("total_retries", 0) >= 0
    assert meta.get("precheck_count", 0) >= 0
    assert meta.get("recovery_count", 0) >= 0
    assert meta.get("execution_risk_score", 0) >= 0
    assert isinstance(meta.get("pipeline_order"), list)
