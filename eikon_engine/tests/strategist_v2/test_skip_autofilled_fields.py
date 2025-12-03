from __future__ import annotations

from eikon_engine.strategist.dom_features import extract_dom_features
from eikon_engine.strategist.strategist_v2 import StrategistV2


class StubPlanner:
    async def create_plan(self, goal: str, last_result=None):  # noqa: D401
        return {"plan_id": "stub", "goal": goal, "tasks": []}


def make_strategist(**kwargs) -> StrategistV2:
    return StrategistV2(planner=StubPlanner(), **kwargs)


def test_skip_autofilled_fill_step() -> None:
    strategist = make_strategist()
    html = """
    <form>
      <input id="email" name="email" value="foo@example.com" />
      <input id="password" type="password" value="secret" />
    </form>
    """
    strategist._last_features = extract_dom_features(html)  # type: ignore[attr-defined]
    step = {
        "step_id": "s2",
        "action_payload": {
            "action": "fill",
            "selector": "#email",
        },
    }
    run_ctx = {"current_url": "https://example.com/login"}
    assert strategist.should_skip_step(run_ctx, step)


def test_do_not_skip_when_not_filled() -> None:
    strategist = make_strategist()
    html = "<input id='email' name='email' value='' />"
    strategist._last_features = extract_dom_features(html)  # type: ignore[attr-defined]
    step = {
        "step_id": "s2",
        "action_payload": {
            "action": "fill",
            "selector": "#email",
        },
    }
    run_ctx = {"current_url": "https://example.com/login"}
    assert not strategist.should_skip_step(run_ctx, step)
