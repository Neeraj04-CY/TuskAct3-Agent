"""DOM feature extraction helpers for Strategist v2."""

from __future__ import annotations

import re
from html import unescape
from typing import Any, Dict, List, Tuple

WhitespacePattern = re.compile(r"\s+")
TAG_RE = re.compile(r"<(?P<tag>\w+)(?P<body>[^>]*)>", re.IGNORECASE)
INPUT_RE = re.compile(r"<input\b([^>]*)>", re.IGNORECASE)
BUTTON_RE = re.compile(r"<button\b([^>]*)>(.*?)</button>", re.IGNORECASE | re.DOTALL)
ANCHOR_RE = re.compile(r"<a\b([^>]*)>(.*?)</a>", re.IGNORECASE | re.DOTALL)
ATTR_RE = re.compile(r"([a-zA-Z_][\w:-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))")
STRIP_TAGS_RE = re.compile(r"<[^>]+>")


DomEntry = Dict[str, Any]
DomFeatures = Dict[str, Any]


def normalize_text(text: str) -> str:
    """Collapse whitespace and strip surrounding spaces."""

    return WhitespacePattern.sub(" ", text or "").strip()


def _parse_attributes(raw: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for key, dbl, sgl, bare in ATTR_RE.findall(raw or ""):
        value = dbl or sgl or bare or ""
        attrs[key.lower()] = unescape(value)
    return attrs


def _build_selector(attrs: Dict[str, str]) -> str:
    if not attrs:
        return ""
    if "id" in attrs and attrs["id"]:
        return f"#{attrs['id']}"
    if "name" in attrs and attrs["name"]:
        return f"[name='{attrs['name']}']"
    if "data-testid" in attrs and attrs["data-testid"]:
        return f"[data-testid='{attrs['data-testid']}']"
    class_value = attrs.get("class")
    if class_value:
        first = class_value.split()[0]
        if first:
            return f".{first}"
    return ""


def _extract_tag_text(html: str) -> str:
    return normalize_text(unescape(STRIP_TAGS_RE.sub(" ", html or "")))


def extract_dom_features(dom: str) -> DomFeatures:
    """Return lightweight structural signals needed by Strategist v2."""

    html = dom or ""
    text = _extract_tag_text(html).lower()
    inputs: List[DomEntry] = []
    for match in INPUT_RE.finditer(html):
        attrs = _parse_attributes(match.group(1))
        selector = _build_selector(attrs)
        entry: DomEntry = {
            "tag": "input",
            "selector": selector,
            "name": attrs.get("name", ""),
            "type": (attrs.get("type") or "text").lower(),
            "value": attrs.get("value", ""),
            "attributes": attrs,
        }
        inputs.append(entry)

    buttons: List[DomEntry] = []
    for attrs_raw, body in BUTTON_RE.findall(html):
        attrs = _parse_attributes(attrs_raw)
        selector = _build_selector(attrs)
        entry = {
            "tag": "button",
            "selector": selector,
            "text": _extract_tag_text(body),
            "attributes": attrs,
        }
        buttons.append(entry)

    links: List[DomEntry] = []
    for attrs_raw, body in ANCHOR_RE.findall(html):
        attrs = _parse_attributes(attrs_raw)
        selector = _build_selector(attrs)
        entry = {
            "tag": "a",
            "selector": selector,
            "text": _extract_tag_text(body),
            "attributes": attrs,
        }
        links.append(entry)

    keywords = set(token for token in text.split() if token)
    input_lookup = {entry["selector"]: entry for entry in inputs if entry["selector"]}
    filled_inputs = {selector: entry for selector, entry in input_lookup.items() if entry.get("value")}

    features: DomFeatures = {
        "inputs": inputs,
        "buttons": buttons,
        "links": links,
        "text": text,
        "keywords": keywords,
        "input_lookup": input_lookup,
        "filled_inputs": filled_inputs,
        "has_password_input": any(entry.get("type") == "password" for entry in inputs),
        "has_email_input": any("email" in (entry.get("name") or "") for entry in inputs),
        "button_texts": [entry.get("text", "").lower() for entry in buttons],
    }
    return features


def selector_in_dom(selector: str, dom: str) -> bool:
    if not selector:
        return True
    if selector.startswith("#"):
        token = selector[1:]
        return token.lower() in dom.lower()
    if selector.startswith("[name="):
        token = selector.replace("[name=", "").strip("']\"").lower()
        return f'name="{token}"' in dom.lower() or f"name='{token}'" in dom.lower()
    if selector.startswith("."):
        token = selector[1:]
        return token.lower() in dom.lower()
    return selector.lower() in dom.lower()


__all__ = [
    "DomFeatures",
    "extract_dom_features",
    "selector_in_dom",
]
