from __future__ import annotations

from eikon_engine.strategist.strategist_v2 import StrategistV2

FAILING_HTML = """
<div class="app">
  <p>Loading failed state</p>
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
                "bucket": "interaction",
                "depends_on": [],
                "inputs": {
                    "actions": [
                        {"id": "s1", "action": "click", "selector": "#cta"},
                    ]
                },
            }
        ],
    }


def make_strategist() -> StrategistV2:
    return StrategistV2(planner=StubPlanner())


def make_failure_result() -> dict:
    return {
        "error": "timeout",
        "completion": {"complete": False, "reason": "failed"},
        "dom_snapshot": FAILING_HTML,
        "meta": {"url": "https://site"},
    }


def test_progressive_recovery_schedules_reload_and_nav() -> None:
    strategist = make_strategist()
    strategist.load_plan(make_plan())
    run_ctx = {"current_url": "https://site"}
    planned_step = strategist.next_step()

    strategist.on_step_result(run_ctx, planned_step.metadata, make_failure_result())
    inserted = strategist.peek_step()
    assert inserted["source"] == "progressive_recovery"
    assert inserted["action_payload"]["action"] == "reload_if_failed"

    # simulate another failure for the original plan step
    strategist._schedule_progressive_recovery(run_ctx, planned_step.metadata)  # type: ignore[attr-defined]
    inserted = strategist.peek_step()
    assert inserted["action_payload"]["action"] in {"navigate", "extract_dom"}
    assert inserted["source"] == "progressive_recovery"

    # third failure forces replan flag
    strategist._schedule_progressive_recovery(run_ctx, planned_step.metadata)  # type: ignore[attr-defined]
    assert run_ctx.get("force_replan") is True
    assert strategist.should_replan(run_ctx, planned_step.metadata, {"meta": {}})
