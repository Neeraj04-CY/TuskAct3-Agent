from __future__ import annotations

import pytest

from eikon_engine.strategist.strategist_v2 import StrategistV2


def make_plan(tag: str) -> dict:
    return {
        "plan_id": tag,
        "goal": tag,
        "tasks": [
            {
                "id": f"{tag}_task",
                "tool": "BrowserWorker",
                "bucket": "demo",
                "depends_on": [],
                "inputs": {"actions": [{"id": f"{tag}_s1", "action": "navigate", "url": "https://site"}]},
            }
        ],
    }


class StubPlanner:
    async def create_plan(self, goal: str, last_result=None):  # noqa: D401
        return make_plan(goal)


def make_run_result() -> dict:
    return {
        "completion": {"complete": True, "reason": "done"},
        "dom_snapshot": "<html></html>",
        "meta": {"url": "https://site"},
    }


def make_payload_only_result() -> dict:
    return {
        "completion": {"complete": True, "reason": "done"},
        "dom_snapshot": "<html></html>",
    }


def test_queue_subgoal_with_inline_plan() -> None:
    strategist = StrategistV2(planner=StubPlanner())
    strategist.load_plan(make_plan("primary"))
    strategist.queue_subgoal("secondary", plan=make_plan("secondary"))
    run_ctx = {"current_url": "https://site"}
    step = strategist.next_step()
    strategist.on_step_result(run_ctx, step.metadata, make_run_result())
    strategist.record_result(make_run_result())
    assert strategist.has_next()
    assert strategist.peek_step()["task_id"] == "secondary_task"


class RecordingPlanner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def create_plan(self, goal: str, last_result=None):  # noqa: D401
        self.calls.append(goal)
        return make_plan(goal)


@pytest.mark.asyncio
async def test_queue_subgoal_triggers_planner() -> None:
    planner = RecordingPlanner()
    strategist = StrategistV2(planner=planner)
    await strategist.initialize("primary")
    strategist.queue_subgoal("secondary")
    run_ctx = {"current_url": "https://site"}
    step = strategist.next_step()
    strategist.on_step_result(run_ctx, step.metadata, make_run_result())
    strategist.record_result(make_run_result())
    await strategist.ensure_plan()
    assert planner.calls == ["primary", "secondary"]
    assert strategist.has_next()
    assert strategist.peek_step()["task_id"] == "secondary_task"


def test_on_step_result_uses_action_payload_url_when_meta_missing() -> None:
    strategist = StrategistV2(planner=StubPlanner())
    strategist.load_plan(make_plan("primary"))
    run_ctx: dict[str, str | None] = {"current_url": None}
    step = strategist.next_step()
    strategist.on_step_result(run_ctx, step.metadata, make_payload_only_result())
    assert run_ctx["current_url"] == "https://site"
