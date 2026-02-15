from __future__ import annotations

from eikon_engine.learning.override_engine import LearningOverrideEngine


def test_override_replaces_low_score_step() -> None:
    plan = [
        {"id": "sg-1", "description": "login", "intent": "login", "skill": None},
        {"id": "sg-2", "description": "navigate", "intent": "navigation", "skill": None},
    ]
    scores = {("unknown", "login", "login"): -0.2, ("unknown", "navigate", "navigation"): 0.5}
    engine = LearningOverrideEngine(scores=scores, preferred_skills=["login_form_skill"], threshold=0.0, hard_floor=-0.6)

    decision = engine.apply_override(plan, {"learning_bias": {"preferred_skills": ["login_form_skill"]}})

    assert decision.decision_type in {"REPLACE_WITH_SKILL", "REORDER"}
    assert decision.adjusted_plan
    assert decision.adjusted_plan[0].get("skill") == "login_form_skill" or decision.decision_type == "REORDER"


def test_override_refuses_on_hard_floor() -> None:
    plan = [{"id": "sg-1", "description": "login", "intent": "login", "skill": None}]
    scores = {("unknown", "login", "login"): -0.9}
    engine = LearningOverrideEngine(scores=scores, preferred_skills=[], threshold=0.0, hard_floor=-0.6)

    decision = engine.apply_override(plan, {})

    assert decision.decision_type == "REFUSE"
