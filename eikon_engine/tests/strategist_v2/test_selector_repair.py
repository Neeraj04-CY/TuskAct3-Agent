from __future__ import annotations

from eikon_engine.strategist.dom_features import extract_dom_features
from eikon_engine.strategist.strategist_v2 import StrategistV2


class StubPlanner:
    async def create_plan(self, goal: str, last_result=None):  # noqa: D401
        return {"plan_id": "stub", "goal": goal, "tasks": []}


def make_strategist(**kwargs):
    return StrategistV2(planner=StubPlanner(), **kwargs)


def make_plan_with_actions(actions: list[dict]) -> dict:
    return {
        "plan_id": "demo",
        "goal": "demo",
        "tasks": [
            {
                "id": "task_1",
                "tool": "BrowserWorker",
                "bucket": "interaction",
                "depends_on": [],
                "inputs": {"actions": actions},
            }
        ],
    }


def test_selector_micro_repair_loose_match() -> None:
    strategist = make_strategist()
    planned_step = {"action_payload": {"action": "click", "selector": "#login-button"}}
    dom = """<button name='login' class='primary'>Log In</button>"""
    features = extract_dom_features(dom)
    repair = strategist.apply_micro_repair(planned_step, features)
    assert repair["patched"] is True
    assert "selector" not in planned_step  # original untouched
    assert repair["new_selector"] != "#login-button"


def test_selector_micro_repair_label_match() -> None:
    strategist = make_strategist()
    planned_step = {"action_payload": {"action": "click", "selector": "#submit-btn", "label": "Log In"}}
    dom = """
    <div>
      <button id="alt-login">Log In</button>
    </div>
    """
    features = extract_dom_features(dom)
    repair = strategist.apply_micro_repair(planned_step, features)
    assert repair["patched"]
    assert repair["reason"] == "label_match"


def test_selector_micro_repair_form_inference() -> None:
    strategist = make_strategist()
    planned_step = {"action_payload": {"action": "fill", "selector": "#user-email"}}
    dom = """
    <form>
      <input name="email" value="" />
      <input type="password" id="pwd" />
    </form>
    """
    features = extract_dom_features(dom)
    repair = strategist.apply_micro_repair(planned_step, features)
    assert repair["patched"]
    assert "email" in repair["new_selector"]


def test_selector_repair_inserts_step_using_artifact(tmp_path) -> None:
    strategist = make_strategist()
    strategist.load_plan(
        make_plan_with_actions([
            {"id": "s1", "action": "fill", "selector": "#user-email"},
        ])
    )
    run_ctx = {"current_url": "https://site/login"}
    step = strategist.next_step()
    artifact_path = tmp_path / "dom.html"
    artifact_path.write_text("<input name='email' value='' />", encoding="utf-8")
    result = {
        "completion": {"complete": False, "reason": "missing selector"},
        "failure_dom_path": str(artifact_path),
        "meta": {"url": "https://site/login"},
        "error": "selector missing",
    }
    strategist.on_step_result(run_ctx, step.metadata, result)
    repair_step = strategist.peek_step()
    assert repair_step["source"] == "micro_repair"
    assert repair_step["action_payload"]["selector"] != "#user-email"
