from __future__ import annotations

from eikon_engine.planning.planner_v3 import inject_prechecks


def test_prechecks_injected_for_low_durability_click() -> None:
    tasks = [
        {
            "id": "task_1",
            "tool": "BrowserWorker",
            "inputs": {"actions": [{"action": "click", "durability": "low"}]},
            "depends_on": [],
            "bucket": "form",
        }
    ]

    updated = inject_prechecks(tasks)
    actions = updated[0]["inputs"]["actions"]

    assert actions[0]["action"] == "wait_for_selector"
    assert actions[0].get("_precheck") is True
    assert actions[1]["action"] == "dom_presence_check"
    assert actions[1].get("_precheck") is True
    assert actions[2]["action"] == "click"
