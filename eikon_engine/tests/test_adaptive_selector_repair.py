from __future__ import annotations

from copy import deepcopy

from eikon_engine.api import llm_repair
from eikon_engine.core.adaptive_controller import AdaptiveController

BASE_PLAN = {
    "plan_id": "demo",
    "goal": "Demo goal",
    "tasks": [
        {
            "id": "task_1",
            "tool": "BrowserWorker",
            "bucket": "navigation",
            "depends_on": [],
            "inputs": {
                "actions": [
                    {"id": "s1", "action": "navigate", "url": "https://example.com"},
                    {"id": "s2", "action": "click", "selector": "#missing"},
                ]
            },
        }
    ],
}


def make_plan() -> dict:
    return deepcopy(BASE_PLAN)


def test_adaptive_selector_repair(monkeypatch) -> None:
    controller = AdaptiveController(max_corrections=2, max_selector_repairs=2)
    plan = make_plan()
    failure_report = {
        "step_id": "s2",
        "error": "selector not found",
        "dom_excerpt": "<button id='login'>",
        "worker_trace": {"step_id": "s2"},
    }

    def fake_request(_report):
        return {
            "type": "patch_selector",
            "payload": {"step_id": "s2", "selector": "#login"},
        }

    monkeypatch.setattr(llm_repair, "request_llm_fix", fake_request)
    assert controller.should_call_llm(failure_report) is True
    delta = controller.propose_fix(failure_report)
    assert delta is not None
    controller.apply_fix(plan, delta)
    patched = plan["tasks"][0]["inputs"]["actions"][1]
    assert patched["selector"] == "#login"
    assert controller.selector_repairs == 1