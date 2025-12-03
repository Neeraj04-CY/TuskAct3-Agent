from __future__ import annotations

import pytest

from eikon_engine.strategist.behavior_learner import BehaviorLearner


def test_predictions_stabilize_after_multiple_episodes() -> None:
    learner = BehaviorLearner()
    fingerprint = "page::integration"
    planner_events = [{"type": "subgoal", "name": "refine_query", "status": "completed"}]

    for _ in range(5):
        learner.update(fingerprint, reward_trace=[{"reward": 0.15}], planner_events=planner_events, repair_events=None)
        learner.predict(fingerprint, None, None)

    first = learner.predict(fingerprint, None, None)
    learner.update(fingerprint, reward_trace=[{"reward": 0.14}], planner_events=planner_events, repair_events=None)
    second = learner.predict(fingerprint, None, None)

    assert abs(second["difficulty"] - first["difficulty"]) < 0.05

    summary = learner.summarize().get(fingerprint)
    assert summary is not None
    assert pytest.approx(summary["last_prediction"]["difficulty"], rel=1e-3) == second["difficulty"]
    assert summary["subgoals"]["refine_query"]["success_rate"] == pytest.approx(1.0)
