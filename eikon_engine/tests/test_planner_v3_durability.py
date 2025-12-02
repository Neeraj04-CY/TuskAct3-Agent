from __future__ import annotations

from eikon_engine.planning.planner_v3 import estimate_durability, plan_from_goal


def test_estimate_durability_levels() -> None:
    context = {"goal_text": "Visit static page"}
    navigation_step = {"action": "navigate", "url": "https://example.com"}
    form_step = {"action": "fill", "selector": "#username", "value": "demo"}
    vague_step = {"action": "click"}

    assert estimate_durability(navigation_step, context) == "high"
    assert estimate_durability(form_step, context) == "medium"
    assert estimate_durability(vague_step, context) == "low"


def test_plan_contains_durability_summary() -> None:
    plan = plan_from_goal("Log in to https://example.com/login", context={})
    summary = plan["meta"].get("durability_summary")

    assert summary is not None
    assert set(summary.keys()) == {"low", "medium", "high"}
    assert sum(summary.values()) >= len(plan["tasks"])
