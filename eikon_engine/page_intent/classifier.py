from __future__ import annotations

from typing import Any, Dict, Optional

from .signals import extract_page_signals
from .types import PageIntent, PageIntentResult

LISTING_URL_TOKENS = ("/companies", "list", "catalog", "results")
LISTING_TEXT_TOKENS = ("companies", "startups", "items", "results", "list")
ARTICLE_TEXT_TOKENS = ("article", "story", "blog", "press")
DETAIL_TOKENS = ("profile", "details", "overview")


class PageIntentClassifier:
    """Deterministic classifier that relies solely on DOM structure and keywords."""

    def classify(
        self,
        dom: str,
        *,
        url: Optional[str] = None,
        signals: Optional[Dict[str, Any]] = None,
        features: Optional[Dict[str, Any]] = None,
    ) -> PageIntentResult:
        metrics = dict(signals or extract_page_signals(dom or ""))
        _ = features  # compatibility hook for existing call sites
        url_hint = (url or "").lower()
        scores: Dict[PageIntent, float] = {intent: 0.0 for intent in PageIntent}

        scores[PageIntent.LOGIN_FORM] = self._login_score(metrics)
        scores[PageIntent.LISTING_PAGE] = self._listing_score(metrics, url_hint, dom)
        scores[PageIntent.ARTICLE_PAGE] = self._article_score(metrics, dom)
        scores[PageIntent.DETAIL_PAGE] = self._detail_score(metrics, dom)
        scores[PageIntent.DASHBOARD] = self._dashboard_score(metrics)
        scores[PageIntent.UNKNOWN] = 0.15

        winner = max(scores.items(), key=lambda item: item[1])
        intent, score = winner
        confidence = min(0.99, max(score, 0.05))
        payload = {**metrics, "url_hint": url_hint, "raw_score": round(score, 3)}
        return PageIntentResult(intent=intent, confidence=round(confidence, 3), signals=payload)

    def _login_score(self, metrics: Dict[str, Any]) -> float:
        password_inputs = metrics.get("password_inputs", 0)
        form_count = metrics.get("form_count", 0)
        if password_inputs == 0:
            return 0.0
        return min(0.95, 0.75 + 0.1 * password_inputs + 0.05 * min(form_count, 2))

    def _listing_score(self, metrics: Dict[str, Any], url_hint: str, dom: str) -> float:
        card_rep = metrics.get("card_repetition", 0)
        list_ratio = metrics.get("list_link_ratio", 0.0)
        link_count = metrics.get("link_count", 0)
        list_bonus = 0.15 if any(token in url_hint for token in LISTING_URL_TOKENS) else 0.0
        text_lower = (dom or "").lower()
        text_hits = sum(1 for token in LISTING_TEXT_TOKENS if token in text_lower)
        base = 0.1 * text_hits + 0.08 * min(card_rep, 5) + 0.12 * list_ratio + 0.02 * min(link_count, 25)
        return min(0.92, base + list_bonus)

    def _article_score(self, metrics: Dict[str, Any], dom: str) -> float:
        article_tags = metrics.get("article_tags", 0)
        paragraph_count = metrics.get("paragraph_count", 0)
        heading_count = metrics.get("heading_count", 0)
        text_lower = (dom or "").lower()
        token_hits = sum(1 for token in ARTICLE_TEXT_TOKENS if token in text_lower)
        base = 0.1 * token_hits + 0.12 * min(article_tags, 3)
        base += 0.02 * min(paragraph_count, 20) + 0.03 * min(heading_count, 6)
        return min(0.85, base)

    def _detail_score(self, metrics: Dict[str, Any], dom: str) -> float:
        heading_count = metrics.get("heading_count", 0)
        table_count = metrics.get("table_count", 0)
        text_lower = (dom or "").lower()
        keyword_hits = sum(1 for token in DETAIL_TOKENS if token in text_lower)
        base = 0.05 * keyword_hits + 0.08 * min(heading_count, 3) + 0.1 * min(table_count, 2)
        return min(0.7, base)

    def _dashboard_score(self, metrics: Dict[str, Any]) -> float:
        nav_score = metrics.get("nav_score", 0)
        card_rep = metrics.get("card_repetition", 0)
        return min(0.8, 0.1 * nav_score + 0.05 * card_rep)

def classify_page_intent(
    dom: str,
    *,
    url: Optional[str] = None,
    signals: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, Any]] = None,
) -> PageIntentResult:
    return _CLASSIFIER.classify(dom, url=url, signals=signals, features=features)


_CLASSIFIER = PageIntentClassifier()


__all__ = ["PageIntentClassifier", "classify_page_intent"]
