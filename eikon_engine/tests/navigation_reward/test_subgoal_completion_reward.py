from __future__ import annotations

from eikon_engine.strategist_v2.navigator_reward_model import compute_reward


def test_subgoal_completion_rewards_bonus() -> None:
    old_dom = "<div><p>start</p></div>"
    new_dom = "<div><p>Dashboard complete</p></div>"
    result = compute_reward(old_dom, new_dom, "https://site", "click", "dashboard")
    assert result["reward"] >= 2.0
    assert any("subgoal_completed" in reason for reason in result["reasons"])
