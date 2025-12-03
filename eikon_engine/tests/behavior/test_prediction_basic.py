from __future__ import annotations

from eikon_engine.strategist.behavior_learner import BehaviorLearner


def test_difficulty_and_repair_signals() -> None:
    learner = BehaviorLearner()
    fingerprint = "page::basic"

    learner.update(
        fingerprint,
        reward_trace=[{"reward": 0.9}, {"reward": 0.7}],
        planner_events=[{"type": "subgoal", "name": "inspect_page", "status": "completed"}],
        repair_events=None,
    )
    easy_prediction = learner.predict(fingerprint, None, None)
    assert easy_prediction["difficulty"] < 0.6

    low_rewards = [{"reward": -0.6}, {"reward": -0.5}, {"reward": -0.4}]
    planner_events = [
        {"type": "subgoal", "name": "collect_inputs", "status": "failed"},
        {"type": "subgoal", "name": "collect_inputs", "status": "failed"},
        {"type": "subgoal", "name": "collect_inputs", "status": "completed"},
    ]
    repairs = [{"patch": {"reason": "selector_healing:label_match"}} for _ in range(3)]
    learner.update(fingerprint, low_rewards, planner_events, repairs)

    hard_prediction = learner.predict(fingerprint, [-0.9, -0.8], repairs)
    assert hard_prediction["difficulty"] > easy_prediction["difficulty"]
    assert hard_prediction["likely_repair"] is True
    assert "collect_inputs" in hard_prediction["recommended_subgoals"]
