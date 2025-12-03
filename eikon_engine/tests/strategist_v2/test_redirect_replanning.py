from __future__ import annotations

from eikon_engine.strategist.strategist_v2 import StrategistV2


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
                "inputs": {"actions": [{"id": "s1", "action": "navigate", "url": "https://site/login"}]},
            }
        ],
    }


def make_strategist() -> StrategistV2:
    return StrategistV2(planner=StubPlanner())


def test_redirect_triggers_replan() -> None:
    strategist = make_strategist()
    strategist.load_plan(make_plan())
    run_ctx = {"current_url": "https://site/login"}
    step = strategist.next_step()
    result = {
        "completion": {"complete": False, "reason": "redirect"},
        "meta": {
            "url": "https://site/login",
            "redirect_url": "https://site/dashboard",
        },
    }
    strategist.on_step_result(run_ctx, step.metadata, result)
    assert strategist.should_replan(run_ctx, step.metadata, result)
