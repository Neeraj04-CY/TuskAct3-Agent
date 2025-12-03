"""Safety guardrails that prevent risky worker actions."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

SENSITIVE_URL_KEYWORDS = {"bank", "auth", "login", "checkout", "wallet"}
SENSITIVE_FIELD_KEYWORDS = {"password", "otp", "token", "secret", "2fa", "ssn"}
RISKY_CLICK_KEYWORDS = {"delete", "remove", "danger", "destroy", "logout", "signout", "reset"}


class SafetyGuardrails:
    def __init__(self, settings: Dict[str, Any] | None = None) -> None:
        self.settings = settings or {}
        extra_sensitive = set(self.settings.get("extra_sensitive_terms", []))
        extra_risky = set(self.settings.get("risky_click_terms", []))
        self.sensitive_keywords = SENSITIVE_FIELD_KEYWORDS | {term.lower() for term in extra_sensitive}
        self.risky_click_keywords = RISKY_CLICK_KEYWORDS | {term.lower() for term in extra_risky}
        self.restricted_domains = {domain.lower() for domain in self.settings.get("restricted_domains", [])}

    def check(self, action: Dict[str, Any], *, current_url: str | None = None) -> Tuple[bool, str | None]:
        kind = (action.get("action") or "").lower()
        if kind == "screenshot":
            return self._check_screenshot(action, current_url=current_url)
        if kind == "click":
            return self._check_click(action)
        return True, None

    def _check_screenshot(self, action: Dict[str, Any], *, current_url: str | None) -> Tuple[bool, str | None]:
        if action.get("allow_sensitive"):
            return True, None
        labels = self._collect_tokens(action, keys=("name", "note", "scope"))
        if any(keyword in labels for keyword in self.sensitive_keywords):
            return False, "screenshot_blocked_sensitive"
        if current_url:
            lowered = current_url.lower()
            domain_hits = self.restricted_domains or set()
            if domain_hits and any(domain in lowered for domain in domain_hits):
                return False, "screenshot_blocked_domain"
            if any(keyword in lowered for keyword in SENSITIVE_URL_KEYWORDS):
                return False, "screenshot_blocked_domain"
        return True, None

    def _check_click(self, action: Dict[str, Any]) -> Tuple[bool, str | None]:
        if action.get("allow_risky") or action.get("force"):
            return True, None
        selector = (action.get("selector") or "").lower()
        labels = self._collect_tokens(action, keys=("label", "text", "name"))
        if any(keyword in selector for keyword in self.risky_click_keywords):
            return False, "click_blocked_risky"
        if any(keyword in labels for keyword in self.risky_click_keywords):
            return False, "click_blocked_risky"
        return True, None

    def _collect_tokens(self, action: Dict[str, Any], *, keys: Iterable[str]) -> str:
        tokens = []
        for key in keys:
            value = action.get(key)
            if isinstance(value, str):
                tokens.append(value.lower())
            elif isinstance(value, (list, tuple)):
                tokens.extend(str(item).lower() for item in value)
        tags = action.get("tags", [])
        tokens.extend(str(tag).lower() for tag in tags if isinstance(tag, (str, int)))
        return " ".join(tokens)


__all__ = ["SafetyGuardrails"]
