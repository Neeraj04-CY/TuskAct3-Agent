from __future__ import annotations

from typing import Any, Dict

from eikon_engine.strategist.strategist_v2 import StrategistV2


class _DummyPlanner:
    async def create_plan(self, goal: str, *, last_result: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return {"tasks": []}

def test_skill_suggestions_merge_into_run_context(monkeypatch) -> None:
    strategist = StrategistV2(planner=_DummyPlanner())

    def fake_suggestions(state: Dict[str, Any], failure: Dict[str, Any] | None) -> Dict[str, Any]:
        return {
            "subgoals": ["capture_dashboard"],
            "repairs": [{"action": "reset_session", "reason": failure["reason"] if failure else ""}],
            "skills": [{"name": "login", "subgoals": 1, "repairs": 1, "metadata": {"description": "test"}}],
        }

    monkeypatch.setattr(strategist, "_skill_registry", type("Registry", (), {"suggestions": staticmethod(fake_suggestions)}))

    run_ctx: Dict[str, Any] = {}
    strategist._apply_skill_suggestions(run_ctx, {"mode": "login_page"}, "unauthorized", {"step_id": "s1"})

    assert "capture_dashboard" in run_ctx.get("suggested_subgoals", [])
    assert run_ctx.get("skill_repair_suggestions")
    assert run_ctx.get("skills")[0]["name"] == "login"
