from __future__ import annotations

from types import MethodType, SimpleNamespace

from eikon_engine.strategist.dom_features import extract_dom_features
from eikon_engine.strategist.strategist_v2 import StrategistV2


class StaticPlanner:
    async def create_plan(self, goal: str, last_result=None):  # noqa: D401
        return make_plan()


def make_plan() -> dict:
    return {
        "plan_id": "behavior-demo",
        "goal": "demo",
        "tasks": [
            {
                "id": "task_1",
                "tool": "BrowserWorker",
                "bucket": "interaction",
                "depends_on": [],
                "inputs": {
                    "actions": [
                        {"id": "step_1", "action": "click", "selector": "#cta"},
                    ]
                },
            }
        ],
    }


def make_strategist() -> StrategistV2:
    strategist = StrategistV2(planner=StaticPlanner())

    def detect_state_stub(self, dom: str, url: str | None):
        return {
            "mode": "form_entry",
            "intent": SimpleNamespace(intent="form_entry", confidence=0.9),
            "features": extract_dom_features(dom),
        }

    strategist.detect_state = MethodType(detect_state_stub, strategist)
    return strategist


def test_behavior_predictions_feed_run_context() -> None:
    strategist = make_strategist()
    strategist.load_plan(make_plan())
    dom_snapshot = """
    <html>
      <body>
        <button id="cta">Continue</button>
      </body>
    </html>
    """
    fingerprint = strategist._page_fingerprint("https://site", dom_snapshot)
    strategist.behavior_learner.update(
        fingerprint,
        reward_trace=[{"reward": -0.7}, {"reward": -0.6}],
        planner_events=[
            {"type": "subgoal", "name": "collect_inputs", "status": "failed"},
            {"type": "subgoal", "name": "collect_inputs", "status": "failed"},
        ],
        repair_events=[{"patch": {"reason": "selector_healing:css"}} for _ in range(2)],
    )

    run_ctx = {"current_url": "https://site", "repair_events": []}
    planned_step = strategist.next_step()
    result = {
        "completion": {"complete": True, "reason": "ok"},
        "dom_snapshot": dom_snapshot,
        "meta": {"url": "https://site"},
    }

    strategist.on_step_result(run_ctx, planned_step.metadata, result)

    predictions = run_ctx.get("behavior_predictions") or []
    assert predictions, "expected predictions to be recorded"
    last_prediction = predictions[-1]
    assert last_prediction["fingerprint"] == fingerprint
    assert run_ctx["behavior_difficulty"] == last_prediction["difficulty"]
    assert "collect_inputs" in run_ctx.get("suggested_subgoals", [])

    planner_events = [entry for entry in run_ctx.get("planner_events", []) if entry.get("type") == "subgoal"]
    statuses = {entry.get("status") for entry in planner_events}
    assert "issued" in statuses