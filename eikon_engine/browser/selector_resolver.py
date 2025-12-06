from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Dict, List, Optional
from urllib.parse import urlparse

USERNAME_TOKENS = ("user", "username", "email", "login", "account", "identifier")
PASSWORD_TOKENS = ("pass", "password", "pwd", "pin", "secret")
BUTTON_TOKENS = ("login", "log in", "sign in", "submit", "continue", "next")
LOGIN_HINT_TOKENS = ("login", "log in", "sign in", "signin")

DEFAULT_LOGIN_SELECTORS: Dict[str, List[str]] = {
    "username": [
        "#username",
        'input[name="username"]',
    ],
    "password": [
        "#password",
        'input[name="password"]',
    ],
    "login_button": [
        'button[type="submit"]',
        ".radius",
    ],
}

HEROKU_LOGIN_SELECTORS: Dict[str, List[str]] = {
    "username": ["#username"],
    "password": ["#password"],
    "login_button": [".radius"],
}

LOGIN_BUNDLE_PRIORITY: List[tuple[str, str]] = [
    ("username", "#username"),
    ("username", 'input[name="username"]'),
    ("password", "#password"),
    ("password", 'input[name="password"]'),
    ("login_button", 'button[type="submit"]'),
    ("login_button", ".radius"),
]


@dataclass
class SelectorCandidate:
    selector: str
    score: float
    metadata: Dict[str, str] = field(default_factory=dict)


class _DomCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: List[Dict[str, str]] = []
        self.buttons: List[Dict[str, str]] = []
        self.labels: Dict[str, str] = {}
        self._active_label: Optional[Dict[str, str]] = None
        self._button_stack: List[Dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:  # type: ignore[override]
        attr_dict = {key: (value or "") for key, value in attrs}
        lowered = tag.lower()
        if lowered == "input":
            self.inputs.append(attr_dict)
            input_type = (attr_dict.get("type") or "").lower()
            if input_type in {"submit", "button"}:
                self.buttons.append({**attr_dict, "text": attr_dict.get("value", "")})
        elif lowered == "label":
            self._active_label = {"for": attr_dict.get("for", ""), "text": ""}
        elif lowered in {"button", "a"}:
            role = (attr_dict.get("role") or "").lower()
            if lowered == "button" or role == "button":
                entry = {**attr_dict, "tag": lowered, "text": ""}
                self._button_stack.append(entry)

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._active_label is not None:
            self._active_label["text"] += data
        if self._button_stack:
            self._button_stack[-1]["text"] += data

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        lowered = tag.lower()
        if lowered == "label" and self._active_label is not None:
            label_for = (self._active_label.get("for") or "").lower()
            if label_for:
                self.labels[label_for] = (self._active_label.get("text") or "").strip()
            self._active_label = None
        elif self._button_stack and lowered == (self._button_stack[-1].get("tag") or lowered):
            entry = self._button_stack.pop()
            entry["text"] = entry.get("text", "").strip()
            self.buttons.append(entry)


class SelectorResolver:
    def __init__(
        self,
        dom: str,
        *,
        mission_text: str | None = None,
        goal_text: str | None = None,
        current_url: str | None = None,
        base_selector: str | None = None,
    ) -> None:
        self.dom = dom or ""
        self.mission_text = (mission_text or "").lower()
        self.goal_text = (goal_text or "").lower()
        self.base_selector = (base_selector or "").lower()
        self.current_url = current_url or ""
        parsed = urlparse(self.current_url) if self.current_url else None
        self._url_host = (parsed.netloc if parsed else "").lower()
        self._url_path = parsed.path if parsed else ""
        parser = _DomCollector()
        try:
            parser.feed(self.dom)
            parser.close()
        except Exception:
            pass
        self.inputs = parser.inputs
        self.buttons = parser.buttons
        self.labels = parser.labels
        self._heroku_login = self._is_heroku_login_url()
        self._login_context = self._detect_login_context()

    def has_login_context(self) -> bool:
        return self._login_context

    def is_heroku_login(self) -> bool:
        return self._heroku_login

    def get_login_selector_bundle(self) -> Dict[str, List[str]]:
        if not self._login_context:
            return {}
        bundle: Dict[str, List[str]] = {role: list(selectors) for role, selectors in DEFAULT_LOGIN_SELECTORS.items()}
        if self._heroku_login:
            for role, selectors in HEROKU_LOGIN_SELECTORS.items():
                existing = bundle.setdefault(role, [])
                for selector in selectors:
                    if selector not in existing:
                        existing.append(selector)
        return bundle

    def login_override_candidates(self) -> List[SelectorCandidate]:
        bundle = self.get_login_selector_bundle()
        if not bundle:
            return []
        candidates: List[SelectorCandidate] = []
        score = 24.0
        seen_pairs: set[tuple[str, str]] = set()
        for role, selector in LOGIN_BUNDLE_PRIORITY:
            if selector in bundle.get(role, []):
                candidates.append(
                    SelectorCandidate(
                        selector=selector,
                        score=round(score, 2),
                        metadata={"source": "login_bundle", "role": role},
                    )
                )
                seen_pairs.add((role, selector))
                score -= 0.5
        for role in ("username", "password", "login_button"):
            for selector in bundle.get(role, []):
                if (role, selector) in seen_pairs:
                    continue
                candidates.append(
                    SelectorCandidate(
                        selector=selector,
                        score=round(max(score, 1.0), 2),
                        metadata={"source": "login_bundle", "role": role},
                    )
                )
                score -= 0.25
        return self._dedupe_candidates(candidates)

    def _detect_login_context(self) -> bool:
        sources = [self.mission_text, self.goal_text, self.base_selector]
        for source in sources:
            if source and any(token in source for token in LOGIN_HINT_TOKENS):
                return True
        return self._heroku_login

    def _is_heroku_login_url(self) -> bool:
        if not self._url_host:
            return False
        return "the-internet.herokuapp.com" in self._url_host and self._url_path.startswith("/login")

    def resolve_username(self) -> List[SelectorCandidate]:
        return self._rank_inputs(USERNAME_TOKENS, preferred_types={"text": 1.0, "email": 1.2})

    def resolve_password(self) -> List[SelectorCandidate]:
        return self._rank_inputs(PASSWORD_TOKENS, preferred_types={"password": 1.5})

    def resolve_login_button(self) -> List[SelectorCandidate]:
        candidates: List[SelectorCandidate] = []
        for button in self.buttons:
            score = self._score_button(button)
            if score <= 0:
                continue
            selector = self._selector_from_attrs(button, default_tag="button")
            if selector:
                candidates.append(SelectorCandidate(selector=selector, score=round(score, 2), metadata={"text": button.get("text", "")}))
        if not candidates:
            fallback = [SelectorCandidate("button[type=submit]", 0.2), SelectorCandidate("button", 0.1)]
            return fallback
        return sorted(self._dedupe_candidates(candidates), key=lambda cand: cand.score, reverse=True)

    def _rank_inputs(self, keywords: tuple[str, ...], preferred_types: Dict[str, float]) -> List[SelectorCandidate]:
        candidates: List[SelectorCandidate] = []
        for attrs in self.inputs:
            score = self._score_input(attrs, keywords, preferred_types)
            if score <= 0:
                continue
            selector = self._selector_from_attrs(attrs)
            if selector:
                candidates.append(SelectorCandidate(selector=selector, score=round(score, 2), metadata=attrs))
        if not candidates:
            return []
        return sorted(self._dedupe_candidates(candidates), key=lambda cand: cand.score, reverse=True)

    def _score_input(self, attrs: Dict[str, str], keywords: tuple[str, ...], preferred_types: Dict[str, float]) -> float:
        score = 0.0
        attr_texts = [attrs.get("id", ""), attrs.get("name", ""), attrs.get("placeholder", ""), attrs.get("aria-label", ""), attrs.get("data-testid", ""), attrs.get("class", "")]
        combined = " ".join(attr_texts).lower()
        for keyword in keywords:
            if keyword in combined:
                score += 3.0
        label_text = self._label_for(attrs.get("id", ""))
        if label_text:
            low = label_text.lower()
            if any(keyword in low for keyword in keywords):
                score += 1.5
        input_type = (attrs.get("type") or "").lower()
        score += preferred_types.get(input_type, 0.0)
        if input_type and input_type not in preferred_types:
            score -= 0.2
        return score

    def _score_button(self, attrs: Dict[str, str]) -> float:
        score = 0.0
        button_text = attrs.get("text", "").lower()
        attr_text = " ".join([
            attrs.get("id", ""),
            attrs.get("name", ""),
            attrs.get("aria-label", ""),
            attrs.get("data-testid", ""),
            button_text,
        ]).lower()
        for keyword in BUTTON_TOKENS:
            if keyword in attr_text:
                score += 3.0
        if attrs.get("type", "").lower() in {"submit", "button"}:
            score += 0.5
        return score

    def _selector_from_attrs(self, attrs: Dict[str, str], *, default_tag: str = "input") -> str:
        element_id = attrs.get("id")
        if element_id:
            return f"#{element_id.strip()}"
        name = attrs.get("name")
        if name:
            return f'{default_tag}[name="{name.strip()}"]'
        data_testid = attrs.get("data-testid")
        if data_testid:
            return f'{default_tag}[data-testid="{data_testid.strip()}"]'
        placeholder = attrs.get("placeholder")
        if placeholder:
            return f'{default_tag}[placeholder="{placeholder.strip()}"]'
        class_attr = attrs.get("class")
        if class_attr:
            first_class = class_attr.split()[0].strip()
            if first_class:
                return f"{default_tag}.{first_class}"
        element_type = attrs.get("type")
        if element_type:
            return f'{default_tag}[type="{element_type.strip()}"]'
        return default_tag

    def _label_for(self, element_id: str | None) -> str:
        if not element_id:
            return ""
        return self.labels.get(element_id.lower(), "")

    def _dedupe_candidates(self, candidates: List[SelectorCandidate]) -> List[SelectorCandidate]:
        seen: Dict[str, SelectorCandidate] = {}
        for candidate in candidates:
            if candidate.selector not in seen or seen[candidate.selector].score < candidate.score:
                seen[candidate.selector] = candidate
        return list(seen.values())


__all__ = [
    "SelectorResolver",
    "SelectorCandidate",
    "USERNAME_TOKENS",
    "PASSWORD_TOKENS",
    "BUTTON_TOKENS",
]
