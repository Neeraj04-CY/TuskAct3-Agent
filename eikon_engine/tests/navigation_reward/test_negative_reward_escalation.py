from __future__ import annotations

from eikon_engine.strategist.strategist_v2 import StrategistV2


PLAN = {
    "plan_id": "demo",
    "goal": "demo",
    "tasks": [
        {
            "id": "task_1",
            "tool": "BrowserWorker",
            "bucket": "nav",
            "depends_on": [],
            "inputs": {
                "actions": [
                    {"id": "s1", "action": "navigate", "url": "https://site"},
                    {"id": "s2", "action": "retry_navigation", "url": "https://site"},
                    {"id": "s3", "action": "retry_navigation_again", "url": "https://site"},
                ]
            },
        }
    ],
}


class StaticPlanner:
    async def create_plan(self, goal: str, last_result=None):  # noqa: D401
        return PLAN


def test_low_reward_escalates_and_requests_replan() -> None:
    strategist = StrategistV2(planner=StaticPlanner())
    strategist.load_plan(PLAN)
    run_ctx = {"current_url": "https://site"}

    first_step = strategist.next_step()
    first_result = {
        "completion": {"complete": True, "reason": "ok"},
        "dom_snapshot": "<div>state</div>",
        "meta": {"url": "https://site"},
    }
    strategist.on_step_result(run_ctx, first_step.metadata, first_result)

    second_step = strategist.next_step()
    second_result = {
        "completion": {"complete": True, "reason": "ok"},
        "dom_snapshot": "<div>state</div>",
        "meta": {"url": "https://site"},
    }
    strategist.on_step_result(run_ctx, second_step.metadata, second_result)

    third_step = strategist.next_step()
    third_result = {
        "completion": {"complete": True, "reason": "ok"},
        "dom_snapshot": "<div>state</div>",
        "meta": {"url": "https://site"},
    }
    strategist.on_step_result(run_ctx, third_step.metadata, third_result)

    last_trace = run_ctx.get("reward_trace", [])[-1]
    assert last_trace["confidence"]["band"] == "low"
    assert run_ctx.get("recovery_severity", 0) >= 1
    assert run_ctx.get("force_replan") is True

