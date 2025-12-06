from __future__ import annotations

from copy import deepcopy

from eikon_engine.api import llm_repair
from eikon_engine.core.adaptive_controller import AdaptiveController

BASE_PLAN = {
    "plan_id": "cookie",
    "goal": "Close popup",
    "tasks": [
        {
            "id": "task_1",
            "tool": "BrowserWorker",
            "bucket": "interaction",
            "depends_on": [],
            "inputs": {
                "actions": [
                    {"id": "s1", "action": "navigate", "url": "https://site"},
                    {"id": "s2", "action": "screenshot", "name": "before.png"},
                ]
            },
        }
    ],
}


def make_plan() -> dict:
    return deepcopy(BASE_PLAN)


def test_adaptive_cookie_popup(monkeypatch) -> None:
    controller = AdaptiveController(max_corrections=3)
    plan = make_plan()
    failure_report = {"step_id": "s2", "error": "cookie banner"}

    responses = [
        {
            "type": "insert_steps",
            "payload": {
                "before_step": "s2",
                "actions": [
                    {
                        "id": "cookie_close",
                        "action": "click",
                        "selector": "button.accept",
                    }
                ],
            },
        }
    ]

    def fake_request(_report):
        return responses.pop(0)

    monkeypatch.setattr(llm_repair, "request_llm_fix", fake_request)
    delta = controller.propose_fix(failure_report)
    plan = controller.apply_fix(plan, delta)
    actions = plan["tasks"][0]["inputs"]["actions"]
    assert actions[1]["id"] == "cookie_close"
    assert actions[2]["id"] == "s2"
