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
                    {"id": "s2", "action": "click", "selector": "#cta"},
                    {"id": "s3", "action": "screenshot", "name": "cta.png"},
                ]
            },
        }
    ],
}


def make_plan() -> dict:
    return deepcopy(BASE_PLAN)


def test_adaptive_navigation_fix(monkeypatch) -> None:
    controller = AdaptiveController()
    plan = make_plan()
    failure_report = {
        "step_id": "s2",
        "error": "stale navigation",
        "dom_excerpt": "",
        "worker_trace": {"step_id": "s2"},
    }

    def fake_request(_report):
        return {
            "type": "navigate",
            "payload": {"before_step": "s2", "url": "https://backup.example"},
        }

    monkeypatch.setattr(llm_repair, "request_llm_fix", fake_request)
    assert controller.should_call_llm(failure_report)
    delta = controller.propose_fix(failure_report)
    plan = controller.apply_fix(plan, delta)
    actions = plan["tasks"][0]["inputs"]["actions"]
    assert actions[1]["action"] == "navigate"
    assert actions[1]["url"] == "https://backup.example"
    assert actions[2]["id"] == "s2"  # original click shifted
*** End***