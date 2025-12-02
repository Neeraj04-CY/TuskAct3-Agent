from __future__ import annotations

from eikon_engine.planning.planner_v3 import plan_from_goal, generate_reflection


def test_reflection_missing_credentials() -> None:
    goal = "Log in to https://example.com/login"
    plan = plan_from_goal(goal, context={})

    reflection = plan["meta"].get("reflection") or {}
    assert "username" in reflection.get("missing_fields", [])
    assert "password" in reflection.get("missing_fields", [])


def test_reflection_conflicting_domains_warning() -> None:
    goal = "Collect info from https://alpha.test and https://beta.test"
    reflection = generate_reflection(goal, [], context={})

    warnings = reflection.get("warnings") or []
    assert warnings and "Multiple distinct domains" in warnings[0]
