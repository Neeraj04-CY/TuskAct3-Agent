from __future__ import annotations

from urllib.parse import urlparse

_HIGH_EXACT = {
    "wikipedia.org",
    "www.wikipedia.org",
    "sec.gov",
    "www.sec.gov",
    "forbes.com",
    "www.forbes.com",
    "bloomberg.com",
    "www.bloomberg.com",
    "crunchbase.com",
    "www.crunchbase.com",
}

_MEDIUM_TOKENS = {
    "news",
    "tech",
    "blog",
    "insight",
    "times",
    "post",
    "journal",
    "reuters",
    "theverge",
    "wired",
}

_LOW_TOKENS = {
    "directory",
    "profile",
    "seo",
    "list",
    "rank",
    "aggregator",
    "db",
}


def _normalize_domain(domain_or_url: str) -> str:
    value = (domain_or_url or "").strip().lower()
    if not value:
        return ""
    if "://" in value:
        value = urlparse(value).netloc.lower()
    return value.split(":")[0]


def rank_source(domain: str) -> float:
    normalized = _normalize_domain(domain)
    if not normalized:
        return 0.0
    if normalized in _HIGH_EXACT:
        return 0.95
    if normalized.endswith(".gov"):
        return 0.93
    if normalized.endswith(".org"):
        return 0.86
    if normalized.count(".") == 1 and normalized.endswith(".com"):
        return 0.88

    medium_hits = sum(1 for token in _MEDIUM_TOKENS if token in normalized)
    low_hits = sum(1 for token in _LOW_TOKENS if token in normalized)
    base = 0.62 + (0.04 * medium_hits) - (0.08 * low_hits)
    return max(0.2, min(0.9, base))


def bucket_source(domain: str) -> str:
    score = rank_source(domain)
    if score >= 0.82:
        return "high_credibility"
    if score >= 0.58:
        return "medium"
    return "low"
