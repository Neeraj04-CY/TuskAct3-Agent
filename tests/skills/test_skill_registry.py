from __future__ import annotations

from eikon_engine.skills.skill_registry import SkillRegistry


def test_skill_registry_discovers_builtin_skills() -> None:
    skills = SkillRegistry.get_all()
    names = {skill.name for skill in skills}
    assert {"form_fill", "login", "extract"}.issubset(names)

    state = {"mode": "login_page", "missing_fields": ["email", "password"]}
    failure = {"reason": "validation error"}
    suggestions = SkillRegistry.suggestions(state, failure)
    assert suggestions["subgoals"], "Expected subgoal suggestions"
    assert suggestions["repairs"], "Expected repair suggestions"
    assert any(entry["name"] == "login" for entry in suggestions["skills"])
