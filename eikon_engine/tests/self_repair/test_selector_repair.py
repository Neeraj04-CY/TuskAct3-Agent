from __future__ import annotations

from eikon_engine.strategist.self_repair import SelfRepairEngine


class DummyStrategist:
    def __init__(self) -> None:
        self.inserted = []
        self._recovery_severity = 0
        self.failure_budget = 3
        self.failure_limit = 3
        self.subgoals = []

    def insert_steps(self, actions, *, bucket, tag):  # noqa: D401 - simple stub
        self.inserted.append({"actions": actions, "bucket": bucket, "tag": tag})

    def queue_subgoal(self, goal: str) -> None:  # noqa: D401
        self.subgoals.append(goal)


def test_selector_repair_generates_patch_and_applies() -> None:
    engine = SelfRepairEngine()
    run_ctx = {"history": []}
    last_action = {
        "step": {
            "step_id": "s1",
            "bucket": "nav",
            "action_payload": {"action": "click", "selector": "#login-btn"},
        },
        "failure": "selector not found",
    }
    patch = engine.analyze_failure(run_ctx, "<html></html>", last_action, -1.0, {"confidence": 0.5, "band": "medium"})
    assert patch is not None
    assert patch["type"] == "selector_update"
    strategist = DummyStrategist()
    engine.apply_patch_to_strategist(strategist, patch)
    assert strategist.inserted
    engine.record_repair_event(run_ctx, patch, {"step_id": "s1"})
    assert run_ctx["repair_events"][0]["patch"]["type"] == "selector_update"
