from __future__ import annotations

from eikon_engine.strategist.strategist_v2 import StrategistV2

OVERLAY_HTML = """
<div class="modal-overlay" role="dialog" aria-modal="true">
  <p>Subscribe to continue</p>
  <button class="close">No thanks</button>
</div>
"""

PLAIN_HTML = """
<div>
  <button id="cta">Continue</button>
</div>
"""


class StubPlanner:
    async def create_plan(self, goal: str, last_result=None):  # noqa: D401
        return make_plan()


def make_plan() -> dict:
    return {
        "plan_id": "demo",
        "goal": "demo",
        "tasks": [
            {
                "id": "task_1",
                "tool": "BrowserWorker",
                "bucket": "navigation",
                "depends_on": [],
                "inputs": {"actions": [{"id": "s1", "action": "navigate", "url": "https://site"}]},
            },
            {
                "id": "task_2",
                "tool": "BrowserWorker",
                "bucket": "interaction",
                "depends_on": ["task_1"],
                "inputs": {
                    "actions": [
                        {"id": "s2", "action": "click", "selector": "#cta"},
                    ]
                },
            },
        ],
    }


def make_strategist() -> StrategistV2:
    return StrategistV2(planner=StubPlanner())


def test_interference_inserts_dismissal() -> None:
    strategist = make_strategist()
    strategist.load_plan(make_plan())
    run_ctx = {"current_url": "https://site"}
    first_step = strategist.next_step()
    result = {
        "completion": {"complete": True, "reason": "navigate"},
        "dom_snapshot": OVERLAY_HTML,
        "meta": {"url": "https://site"},
    }
    strategist.on_step_result(run_ctx, first_step.metadata, result)
    next_step = strategist.peek_step()
    assert next_step["source"] == "interference"
    assert next_step["action_payload"]["action"] == "click"
    assert next_step["action_payload"]["selector"].startswith("#") or next_step["action_payload"]["selector"].startswith(".")


def test_no_interference_when_dom_clean() -> None:
    strategist = make_strategist()
    strategist.load_plan(make_plan())
    run_ctx = {"current_url": "https://site"}
    first_step = strategist.next_step()
    result = {
        "completion": {"complete": True, "reason": "navigate"},
        "dom_snapshot": PLAIN_HTML,
        "meta": {"url": "https://site"},
    }
    strategist.on_step_result(run_ctx, first_step.metadata, result)
    next_step = strategist.peek_step()
    assert next_step["source"] != "interference"
