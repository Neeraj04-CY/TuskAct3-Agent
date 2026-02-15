from __future__ import annotations

import re
from urllib.parse import urlparse

_GENERIC_FOUNDER_WORDS = {
    "fund",
    "former",
    "contact",
    "company",
    "profile",
    "coverage",
    "about",
    "team",
    "founder",
    "cofounder",
    "investors",
    "products",
    "privacy",
    "policy",
}

_INVALID_FOUNDER_PHRASES = {
    "former co",
    "contact info",
}

_REJECT_ENTITY_TERMS = {
    "fund",
    "capital",
    "group",
}

_NON_PERSON_NAME_TOKENS = {
    "market",
    "intelligence",
    "find",
    "monitor",
    "lumonic",
    "how",
    "we",
    "data",
    "platform",
    "getting",
    "use",
    "cases",
    "deal",
    "execution",
    "networking",
    "diligence",
    "fundraising",
    "benchmarking",
    "business",
    "development",
    "asset",
    "allocation",
    "portfolio",
    "companies",
    "deals",
    "funds",
    "credit",
    "debt",
    "lenders",
    "partners",
    "service",
    "providers",
    "overview",
    "desktop",
    "mobile",
    "integrations",
    "integration",
    "capabilities",
    "started",
    "tour",
    "profiles",
    "trial",
    "compare",
    "pricing",
    "partnerships",
    "events",
    "press",
    "center",
    "customer",
    "success",
    "awards",
    "global",
    "league",
    "table",
    "careers",
    "learn",
    "core",
    "leadership",
}

_MARKETING_PHRASES = {
    "click here",
    "learn more",
    "sign up",
    "try now",
    "trusted by",
    "book a demo",
    "get started",
}

_SOCIAL_HOSTS = {
    "linkedin.com",
    "www.linkedin.com",
    "twitter.com",
    "x.com",
    "facebook.com",
    "instagram.com",
    "youtube.com",
    "www.youtube.com",
}


def validate_founders(candidates: list[str], occurrence_counter: dict[str, int] | None = None) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        value = " ".join((raw or "").strip().split())
        if not value:
            continue
        if len(value) < 3:
            continue
        lowered = value.lower()
        if any(phrase in lowered for phrase in _INVALID_FOUNDER_PHRASES):
            continue
        if re.search(r"(?:\b[A-Z]{2,}\b\s*){5,}", value):
            continue
        if any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in _GENERIC_FOUNDER_WORDS):
            continue
        if any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in _REJECT_ENTITY_TERMS):
            continue
        if " " not in value.strip():
            continue
        tokens = [token.lower() for token in value.split() if token]
        if any(token in _NON_PERSON_NAME_TOKENS for token in tokens):
            continue
        if not re.fullmatch(r"[A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){1,3}", value):
            continue
        normalized = _normalize_founder_name(value)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(value)
        if occurrence_counter is not None:
            occurrence_counter[normalized] = occurrence_counter.get(normalized, 0) + 1
    return cleaned


def _normalize_founder_name(value: str) -> str:
    cleaned = re.sub(r"[\.,]", " ", value or "")
    cleaned = " ".join(cleaned.strip().split())
    if not cleaned:
        return ""
    tokens = [token for token in cleaned.split(" ") if token]
    title_tokens = {"mr", "mrs", "ms", "dr", "prof", "sir", "madam", "ceo", "cto", "cfo", "founder", "cofounder", "co-founder"}
    normalized_tokens = [token.lower() for token in tokens if token.lower() not in title_tokens]
    return " ".join(normalized_tokens).strip()


def validate_website(candidates: list[str], preferred_domain_token: str | None = None) -> str:
    preferred = (preferred_domain_token or "").lower().strip()
    valid: list[str] = []
    for item in candidates:
        url = (item or "").strip()
        if not url:
            continue
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if parsed.scheme not in {"http", "https"}:
            continue
        if not host:
            continue
        if any(host == social or host.endswith("." + social) for social in _SOCIAL_HOSTS):
            continue
        valid.append(url)
    if not valid:
        return ""
    if preferred:
        for url in valid:
            if preferred in (urlparse(url).netloc or "").lower():
                return url
    return valid[0]


def clean_description(text: str) -> str:
    value = " ".join((text or "").strip().split())
    if not value:
        return ""
    lowered = value.lower()
    for phrase in _MARKETING_PHRASES:
        lowered = lowered.replace(phrase, "")
    value = lowered.strip().capitalize()
    sentences = re.split(r"(?<=[.!?])\s+", value)
    trimmed = [sentence.strip() for sentence in sentences if sentence.strip()]
    return " ".join(trimmed[:3])


def dedupe_mentions(mentions: list[dict[str, str]]) -> list[dict[str, str]]:
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for mention in mentions:
        title = (mention.get("title") or "").strip()
        url = (mention.get("url") or "").strip()
        if not url:
            continue
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append({"title": title or url, "url": url})
    return unique
