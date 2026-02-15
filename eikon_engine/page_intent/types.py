from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class PageIntent(str, Enum):
    """Canonical intents used to steer strategist behavior."""

    LOGIN_FORM = "login_form"
    LISTING_PAGE = "listing_page"
    DETAIL_PAGE = "detail_page"
    ARTICLE_PAGE = "article_page"
    DASHBOARD = "dashboard"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PageIntentResult:
    """Container that captures the classifier verdict plus supporting signals."""

    intent: PageIntent
    confidence: float
    signals: Dict[str, Any]

    def as_payload(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": round(float(self.confidence), 3),
            "signals": dict(self.signals),
        }


__all__ = ["PageIntent", "PageIntentResult"]
