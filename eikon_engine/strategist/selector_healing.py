from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Tuple

# Minimal element type used across project tests & strategist stubs
Element = Dict[str, object]


@dataclass
class HealingEntry:
    selector: str
    reason: str
    confidence: float
    original_selector: str


def sequence_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def token_similarity(a: str, b: str) -> float:
    a_tokens = {t for t in a.lower().split() if t}
    b_tokens = {t for t in b.lower().split() if t}
    if not a_tokens or not b_tokens:
        return 0.0
    inter = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(inter) / max(1, len(union))


def text_similarity(a: str, b: str) -> float:
    # Combine sequence ratio and token overlap for robustness
    return max(sequence_ratio(a, b), token_similarity(a, b))


def extract_clickables(dom: Iterable[Element]) -> List[Element]:
    # Expect dom elements to have 'clickable' bool, 'selector', 'text', 'tag', 'role'
    return [el for el in dom if bool(el.get("clickable"))]


def _best_by_text(elements: Iterable[Element], target_text: str) -> Tuple[Optional[Element], float]:
    best = None
    best_score = 0.0
    if not target_text:
        return None, 0.0
    for el in elements:
        text = (el.get("text") or "").strip()
        score = text_similarity(text, target_text)
        if score > best_score:
            best_score = score
            best = el
    return best, best_score


def _role_candidates(elements: Iterable[Element], expected_role: Optional[str]) -> List[Element]:
    if not expected_role:
        return []
    exp = expected_role.lower()
    return [el for el in elements if (str(el.get("role") or "").lower() == exp or str(el.get("tag") or "").lower() == exp)]


def _downgrade_selector_by_type(broken_selector: str) -> Optional[str]:
    # Simple heuristic: strip id/class suffixes to reach a generic selector
    # e.g. button#submit-123 -> button[type=submit] (best-effort)
    parts = broken_selector.split()
    if not parts:
        return None
    first = parts[-1]
    # If contains '#' or '.', try to return tag
    if '#' in first:
        tag = first.split('#')[0]
        if tag:
            return tag
        return None
    if '.' in first:
        tag = first.split('.')[0]
        if tag:
            return tag
        return None
    return None


def nearest_clickable_fallback(dom: Iterable[Element], target_text: Optional[str]) -> Optional[Tuple[Element, float]]:
    clickables = extract_clickables(dom)
    if not clickables:
        return None
    if target_text:
        best, score = _best_by_text(clickables, target_text)
        if best:
            return best, score
    # No text or low match — pick the first clickable as a last resort
    return clickables[0], 0.25


def heal_selector(dom: Iterable[Element], broken_selector: str, intent: Optional[Dict] = None) -> Optional[HealingEntry]:
    """
    Attempt to produce a healed selector for `broken_selector`.
    `dom` is an iterable containing elements of the shape used throughout the repo tests:
      { "selector": str, "tag": str, "text": str, "clickable": bool, "role": str }
    `intent` is optional and may contain {"text": "...", "role": "button" } to guide matching.
    Returns HealingEntry or None if no reasonable candidate found.
    """
    # Normalize inputs
    target_text = None
    expected_role = None
    if intent:
        target_text = intent.get("text")
        expected_role = intent.get("role")

    clickables = extract_clickables(dom)

    # 1) Role-based fallback: if intent says role=button, prefer tag=button
    if expected_role:
        candidates = _role_candidates(clickables, expected_role)
        if candidates:
            cand = candidates[0]
            return HealingEntry(selector=cand["selector"], reason="role_fallback", confidence=0.60, original_selector=broken_selector)

    # 2) Fuzzy text matching
    if target_text:
        best, score = _best_by_text(clickables, target_text)
        if best and score > 0.55:
            return HealingEntry(selector=best["selector"], reason="fuzzy_text_match", confidence=float(score), original_selector=broken_selector)

    # 3) Attempt downgrade of broken selector to generic tag/type
    downgraded = _downgrade_selector_by_type(broken_selector)
    if downgraded:
        for el in clickables:
            if str(el.get("tag") or "").lower() == downgraded.lower():
                return HealingEntry(selector=el["selector"], reason="downgrade_fallback", confidence=0.5, original_selector=broken_selector)

    # 4) Nearest-clickable fallback (best-effort)
    nf = nearest_clickable_fallback(dom, target_text)
    if nf:
        el, score = nf
        reason = "nearest_clickable_fallback" if target_text else "first_clickable_fallback"
        return HealingEntry(selector=el["selector"], reason=reason, confidence=float(score), original_selector=broken_selector)

    # Nothing found
    return None
