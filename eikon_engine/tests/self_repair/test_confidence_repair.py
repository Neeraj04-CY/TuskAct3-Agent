from __future__ import annotations

from eikon_engine.strategist.self_repair import SelfRepairEngine


class DummyStrategist:
    def __init__(self) -> None:
        self._recovery_severity = 0
        self.failure_budget = 3
        self.failure_limit = 3
        self.inserted = []
        self.subgoals = []

    def insert_steps(self, actions, *, bucket, tag):  # noqa: D401
        self.inserted.append({"actions": actions, "bucket": bucket, "tag": tag})

    def queue_subgoal(self, goal: str) -> None:  # noqa: D401
        self.subgoals.append(goal)


def make_low_confidence_trace() -> list[dict[str, object]]:
    trace = []
    for idx in range(3):
        trace.append({
            "step_id": f"s{idx}",
            "reward": -0.2,
            "confidence": {"confidence": 0.1, "band": "low"},
            "reasons": ["dom_static:-1.0"],
        })
    return trace


def test_low_confidence_triggers_strategy_patch() -> None:
    engine = SelfRepairEngine()
    run_ctx = {"reward_trace": make_low_confidence_trace()}
    last_action = {
        "step": {"step_id": "s10", "bucket": "nav", "action_payload": {"action": "navigate", "selector": "#cta"}},
        "failure": "",
    }
    patch = engine.analyze_failure(run_ctx, "<html></html>", last_action, -1.5, {"confidence": 0.1, "band": "low"})
    assert patch is not None
    assert patch["type"] == "strategy_param"
    strategist = DummyStrategist()
    before = strategist.failure_budget
    engine.apply_patch_to_strategist(strategist, patch)
    assert strategist.failure_budget == before - 1
    engine.record_repair_event(run_ctx, patch, {"step_id": "s10"})
    assert run_ctx["repair_events"][0]["patch"]["type"] == "strategy_param"
