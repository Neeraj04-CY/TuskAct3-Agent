"""Lightweight heuristics to infer the page intent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .dom_features import DomFeatures, extract_dom_features


@dataclass
class PageIntent:
    intent: str
    confidence: float
    signals: Dict[str, float]


def classify_page_intent(dom: str, *, url: Optional[str] = None, features: Optional[DomFeatures] = None) -> PageIntent:
    html = dom or ""
    features = features or extract_dom_features(html)
    text = features.get("text", "")
    keywords = features.get("keywords", set())
    signals: Dict[str, float] = {}

    if features.get("has_password_input"):
        signals["password"] = 1.0
        return PageIntent(intent="auth_form", confidence=0.95, signals=signals)

    input_count = len(features.get("inputs", []))
    button_texts = features.get("button_texts", [])
    if input_count >= 3 and any(token in btn for btn in button_texts for token in {"submit", "continue", "next"}):
        signals["inputs"] = min(1.0, input_count / 5.0)
        signals["cta"] = 0.7
        return PageIntent(intent="form_entry", confidence=0.85, signals=signals)

    if "search" in keywords and len(features.get("links", [])) >= 5:
        signals["search_keyword"] = 0.8
        signals["links"] = min(1.0, len(features.get("links", [])) / 10.0)
        return PageIntent(intent="search_results", confidence=0.8, signals=signals)

    if any(token in text for token in {"checkout", "cart", "billing"}):
        signals["commerce"] = 0.9
        return PageIntent(intent="checkout", confidence=0.9, signals=signals)

    if len(features.get("links", [])) >= 8:
        signals["links"] = min(1.0, len(features.get("links", [])) / 12.0)
        return PageIntent(intent="listing", confidence=0.7, signals=signals)

    signals["default"] = 0.3
    return PageIntent(intent="unknown", confidence=0.3, signals=signals)


__all__ = ["PageIntent", "classify_page_intent"]
