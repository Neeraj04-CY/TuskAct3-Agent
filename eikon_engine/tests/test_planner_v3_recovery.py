from __future__ import annotations

from eikon_engine.planning.planner_v3 import inject_recovery_steps


def test_recovery_steps_appended() -> None:
    tasks = [
        {
            "id": "task_1",
            "tool": "BrowserWorker",
            "inputs": {
                "actions": [
                    {"action": "navigate", "url": "https://example.com", "durability": "high"},
                    {"action": "fill", "selector": "#username", "durability": "medium"},
                ]
            },
            "depends_on": [],
            "bucket": "navigation",
        }
    ]

    updated = inject_recovery_steps(tasks)

    assert len(updated) == 3  # original + retry + reload tasks
    retry_actions = [task["inputs"]["actions"][0] for task in updated if task["inputs"]["actions"][0]["action"] == "retry"]
    reload_actions = [task["inputs"]["actions"][0] for task in updated if task["inputs"]["actions"][0]["action"] == "reload_if_failed"]

    assert retry_actions and retry_actions[0].get("_recovery") is True
    assert reload_actions and reload_actions[0].get("_recovery") is True
