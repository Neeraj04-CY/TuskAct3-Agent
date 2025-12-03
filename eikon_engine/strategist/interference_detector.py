"""Heuristics for detecting blocking overlays and popups."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .dom_features import DomFeatures, extract_dom_features

OVERLAY_HINTS = {
    "modal",
    "overlay",
    "popup",
    "interstitial",
    "newsletter",
    "subscribe",
    "blocker",
    "announcement",
}
DISMISS_HINTS = {
    "close",
    "dismiss",
    "no thanks",
    "maybe later",
    "skip",
    "got it",
    "allow",
    "deny",
    "accept",
    "decline",
    "continue",
    "back to site",
}
OVERLAY_TAG_RE = re.compile(r"<(?P<tag>\w+)(?P<body>[^>]*)>", re.IGNORECASE)
ATTR_RE = re.compile(r"([a-zA-Z_][\w:-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))")


@dataclass
class InterferenceFinding:
    selector: str
    reason: str
    text: Optional[str] = None
    metadata: Dict[str, Any] | None = None


def detect_interference(dom: str, features: Optional[DomFeatures] = None) -> List[InterferenceFinding]:
    """Return actionable interference dismissals, if any."""

    html = dom or ""
    if not html.strip():
        return []
    lowered = html.lower()
    if not any(keyword in lowered for keyword in OVERLAY_HINTS):
        return []
    features = features or extract_dom_features(dom)
    findings: List[InterferenceFinding] = []
    button = _find_dismiss_button(features)
    overlay_selector = _find_overlay_selector(html)
    selector = (button or {}).get("selector") or overlay_selector
    if selector:
        reason = "dismiss_overlay" if button else "hide_overlay"
        findings.append(
            InterferenceFinding(
                selector=selector,
                reason=reason,
                text=(button or {}).get("text"),
                metadata={"overlay_selector": overlay_selector, "button_selector": (button or {}).get("selector")},
            )
        )
    return findings


def _find_dismiss_button(features: DomFeatures) -> Optional[Dict[str, Any]]:
    for entry in features.get("buttons", []):
        text = (entry.get("text") or "").lower()
        if any(hint in text for hint in DISMISS_HINTS):
            return entry
        attrs = entry.get("attributes", {})
        aria = (attrs.get("aria-label") or "").lower()
        if aria and any(hint in aria for hint in DISMISS_HINTS):
            return entry
    return None


def _find_overlay_selector(html: str) -> Optional[str]:
    for match in OVERLAY_TAG_RE.finditer(html):
        attrs = _parse_attrs(match.group("body"))
        class_val = attrs.get("class", "").lower()
        if any(keyword in class_val for keyword in OVERLAY_HINTS):
            return _selector_from_attrs(attrs)
        data_role = attrs.get("role", "").lower()
        if data_role in {"dialog", "alertdialog"}:
            return _selector_from_attrs(attrs)
        aria_modal = attrs.get("aria-modal", "").lower()
        if aria_modal == "true":
            return _selector_from_attrs(attrs)
    return None


def _parse_attrs(raw: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for key, dbl, sgl, bare in ATTR_RE.findall(raw or ""):
        attrs[key.lower()] = (dbl or sgl or bare or "").strip()
    return attrs


def _selector_from_attrs(attrs: Dict[str, str]) -> Optional[str]:
    if attrs.get("id"):
        return f"#{attrs['id']}"
    class_val = attrs.get("class", "").split()
    if class_val:
        return f".{class_val[0]}"
    data_testid = attrs.get("data-testid")
    if data_testid:
        return f"[data-testid='{data_testid}']"
    return None


__all__ = ["InterferenceFinding", "detect_interference"]
