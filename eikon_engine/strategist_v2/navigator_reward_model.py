"""Reward model for real-world navigation progress."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from eikon_engine.strategist.page_intent import classify_page_intent

OVERLAY_KEYWORDS = {"modal", "overlay", "popup", "newsletter", "subscribe", "accept cookies", "consent"}
LOOP_KEYWORDS = {"retry", "reload", "again", "loop", "repeat"}


def compute_reward(old_dom: str | None, new_dom: str | None, url: str | None, action: str | None, subgoal: str | None) -> Dict[str, object]:
    """Return reward + rationale for the strategist."""

    previous = _normalize_dom(old_dom)
    current = _normalize_dom(new_dom)
    reward = 0.0
    reasons: List[str] = []

    similarity = _dom_similarity(previous, current)
    changed = similarity < 0.92
    if changed:
        reward += 0.5
        reasons.append("dom_changed:+0.5")
    else:
        reward -= 1.0
        reasons.append("dom_static:-1.0")

    overlays = _has_overlay(current)
    if overlays:
        reward -= 1.5
        reasons.append("overlay_detected:-1.5")

    progressed, completed = _intent_alignment(previous, current, subgoal)
    if progressed:
        reward += 1.0
        reasons.append("intent_progress:+1.0")
    if completed:
        reward += 2.0
        reasons.append("subgoal_completed:+2.0")

    if not changed and _looks_like_loop(action, similarity):
        reward -= 2.0
        reasons.append("loop_detected:-2.0")

    return {"reward": round(reward, 3), "reasons": reasons}


def _normalize_dom(dom: str | None) -> str:
    return (dom or "").strip()


def _dom_similarity(old_dom: str, new_dom: str) -> float:
    if not old_dom and not new_dom:
        return 1.0
    window_old = old_dom[:2000]
    window_new = new_dom[:2000]
    return SequenceMatcher(None, window_old, window_new).ratio()


def _has_overlay(dom: str) -> bool:
    lowered = dom.lower()
    return any(keyword in lowered for keyword in OVERLAY_KEYWORDS)


def _intent_alignment(old_dom: str, new_dom: str, subgoal: str | None) -> Tuple[bool, bool]:
    progressed = False
    completed = False
    subgoal_lower = (subgoal or "").lower().strip()
    if subgoal_lower and subgoal_lower in new_dom.lower():
        completed = True
    new_intent = classify_page_intent(new_dom or "")
    old_intent = classify_page_intent(old_dom or "") if old_dom else None
    if not completed and subgoal_lower and subgoal_lower.split()[0] in new_intent.intent:
        progressed = True
    if not completed and old_intent and new_intent.intent != "unknown" and new_intent.intent != old_intent.intent:
        progressed = True
    return progressed, completed


def _looks_like_loop(action: str | None, similarity: float) -> bool:
    label = (action or "").lower()
    return similarity > 0.97 and any(keyword in label for keyword in LOOP_KEYWORDS)


__all__ = ["compute_reward"]
