"""Browser state classification utilities for Strategist v2."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .dom_features import DomFeatures, extract_dom_features
from eikon_engine.page_intent import classify_page_intent

COOKIE_KEYWORDS = {"cookie", "consent", "gdpr"}
LOGIN_KEYWORDS = {"login", "log in", "sign in"}
DASHBOARD_KEYWORDS = {"dashboard", "account", "profile", "workspace", "projects"}
ERROR_KEYWORDS = {"error", "not found", "forbidden", "unavailable", "failed", "404", "500"}


def detect_page_mode(dom: str, url: Optional[str] = None) -> str:
    features = extract_dom_features(dom)
    if is_error_page(dom, url):
        return "error_page"
    if find_cookie_popup(dom, features):
        return "cookie_popup"
    if is_login_page(features):
        return "login_page"
    if is_dashboard_page(features):
        return "dashboard_page"
    return "unknown"


def is_login_page(features: DomFeatures) -> bool:
    buttons = features.get("button_texts", [])
    text = features.get("text", "")
    has_login_text = any(keyword in text for keyword in LOGIN_KEYWORDS)
    has_password = bool(features.get("has_password_input"))
    has_username = bool(features.get("has_email_input")) or any("user" in (entry.get("name") or "") for entry in features.get("inputs", []))
    has_login_button = any(any(keyword in btn for keyword in {"login", "sign"}) for btn in buttons)
    return has_password and (has_username or has_login_text or has_login_button)


def is_dashboard_page(features: DomFeatures) -> bool:
    if features.get("has_password_input"):
        return False
    text = features.get("text", "")
    if any(keyword in text for keyword in DASHBOARD_KEYWORDS):
        return True
    nav_like = any("nav" in (entry.get("attributes", {}).get("class", "")) for entry in features.get("links", []))
    tiles = sum(1 for entry in features.get("buttons", []) if "card" in (entry.get("attributes", {}).get("class", "")))
    return bool(nav_like or tiles >= 2)


def is_error_page(dom: str, url: Optional[str] = None) -> bool:
    text = (dom or "").lower()
    if url and "error" in url.lower():
        return True
    return any(keyword in text for keyword in ERROR_KEYWORDS)


def find_cookie_popup(dom: str, features: Optional[DomFeatures] = None) -> Optional[Dict[str, Any]]:
    html_text = (dom or "").lower()
    if not any(keyword in html_text for keyword in COOKIE_KEYWORDS):
        return None
    features = features or extract_dom_features(dom)
    for button in features.get("buttons", []):
        text = (button.get("text") or "").lower()
        if any(word in text for word in {"accept", "agree"}):
            selector = button.get("selector") or "button"
            return {"selector": selector, "text": button.get("text", "Accept Cookies")}
    return None


def detect_state(dom: str, url: Optional[str] = None) -> Dict[str, Any]:
    features = extract_dom_features(dom)
    if is_error_page(dom, url):
        mode = "error_page"
    elif find_cookie_popup(dom, features):
        mode = "cookie_popup"
    elif is_login_page(features):
        mode = "login_page"
    elif is_dashboard_page(features):
        mode = "dashboard_page"
    else:
        mode = "unknown"
    intent = classify_page_intent(dom, url=url, features=features)
    return {"mode": mode, "features": features, "intent": intent}


__all__ = [
    "detect_page_mode",
    "detect_state",
    "find_cookie_popup",
    "is_dashboard_page",
    "is_error_page",
    "is_login_page",
]
