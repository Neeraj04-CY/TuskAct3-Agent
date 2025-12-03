from __future__ import annotations

from eikon_engine.strategist.self_repair import SelfRepairEngine


class DummyStrategist:
    def __init__(self) -> None:
        self.inserted = []
        self._recovery_severity = 0
        self.failure_budget = 4
        self.failure_limit = 4
        self.subgoals = []

    def insert_steps(self, actions, *, bucket, tag):  # noqa: D401
        self.inserted.append({"actions": actions, "bucket": bucket, "tag": tag})

    def queue_subgoal(self, goal: str) -> None:  # noqa: D401
        self.subgoals.append(goal)


def make_reward_trace(values):
    trace = []
    for idx, value in enumerate(values, start=1):
        trace.append({
            "step_id": f"s{idx}",
            "reward": value,
            "confidence": {"confidence": 0.4, "band": "medium"},
            "reasons": [],
        })
    return trace


def test_reward_stagnation_generates_subgoal_patch() -> None:
    engine = SelfRepairEngine()
    run_ctx = {"reward_trace": make_reward_trace([0.05, 0.06, 0.05])}
    last_action = {
        "step": {"step_id": "s3", "bucket": "nav", "action_payload": {"action": "click", "selector": "#cta"}},
        "failure": "",
    }
    patch = engine.analyze_failure(run_ctx, "<html></html>", last_action, 0.05, {"confidence": 0.4, "band": "medium"})
    assert patch is not None
    assert patch["type"] == "subgoal"
    strategist = DummyStrategist()
    engine.apply_patch_to_strategist(strategist, patch)
    assert strategist.subgoals


def test_per_step_limit_enforced() -> None:
    engine = SelfRepairEngine()
    run_ctx = {"reward_trace": make_reward_trace([0.9, 0.91, 0.9])}
    last_action = {
        "step": {"step_id": "s5", "bucket": "nav", "action_payload": {"action": "click", "selector": "#cta"}},
        "failure": "selector missing",
    }
    first = engine.analyze_failure(run_ctx, "<html></html>", last_action, 0.0, {"confidence": 0.5, "band": "medium"})
    second = engine.analyze_failure(run_ctx, "<html></html>", last_action, 0.0, {"confidence": 0.5, "band": "medium"})
    third = engine.analyze_failure(run_ctx, "<html></html>", last_action, 0.0, {"confidence": 0.5, "band": "medium"})
    assert first is not None and second is not None
    assert third is None  # capped at 2 attempts per step
