from __future__ import annotations

from eikon_engine.strategist_v2.navigator_reward_model import compute_reward


def test_dom_change_increases_reward() -> None:
    old_dom = "<div><p>old</p></div>"
    new_dom = "<div><p>new content added</p></div>"
    result = compute_reward(old_dom, new_dom, "https://site", "navigate", None)
    assert result["reward"] > 0
    assert any("dom_changed" in reason for reason in result["reasons"])


def test_dom_similarity_penalized() -> None:
    dom = "<div><p>same</p></div>"
    result = compute_reward(dom, dom, "https://site", "retry", None)
    assert result["reward"] <= -1.0
    assert any("dom_static" in reason for reason in result["reasons"])
