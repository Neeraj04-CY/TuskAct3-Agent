from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from ..base import Skill

_BLOCK_CARD_FRAGMENT = re.compile(
    r"<(?P<tag>div|li|article|section)(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)
_ANCHOR_CARD_PATTERN = re.compile(r"<a(?P<attrs>[^>]*)>(?P<body>.*?)</a>", re.IGNORECASE | re.DOTALL)
_HEADING_PATTERN = re.compile(r"<h[1-4][^>]*>(?P<text>.*?)</h[1-4]>", re.IGNORECASE | re.DOTALL)
_NAME_PATTERN = re.compile(
    r"<(?:span|div|p)[^>]*class=\"[^\"]*(?:co[-_]?name|company[-_ ]?name|listing[-_ ]?name|name)[^\"]*\"[^>]*>(?P<text>.*?)</(?:span|div|p)>",
    re.IGNORECASE | re.DOTALL,
)
_LINK_PATTERN = re.compile(r"<a[^>]+href=\"(?P<href>[^\"]+)\"[^>]*>(?P<label>.*?)</a>", re.IGNORECASE | re.DOTALL)
_ATTR_LINK_PATTERN = re.compile(r"href\s*=\s*['\"](?P<href>[^'\"]+)['\"]", re.IGNORECASE)
_FOUNDERS_PATTERN = re.compile(r"founders?[:\-]\s*(?P<names>[^<\n]+)", re.IGNORECASE)
_CARD_CLASS_HINTS = ("card", "company", "result", "item", "startup", "listing")


class ListingExtractionSkill(Skill):
    name = "listing_extraction_skill"

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        html = context.get("html")
        page = context.get("page")
        if html is None and page is not None:
            html = await page.content()
        if not html:
            return {"status": "failed", "reason": "missing_dom"}
        page_url = context.get("page_url")
        if not page_url and page is not None:
            page_url = getattr(page, "url", None)
        cards = self._extract_cards(html, page_url)
        if not cards:
            return {"status": "failed", "reason": "no_cards"}
        selected = self._normalize_card(self._select_card(cards), page_url)
        artifact_path = context.get("artifact_path")
        if artifact_path:
            path = Path(artifact_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
        return {
            "status": "success",
            "result": selected,
            "items_found": len(cards),
        }

    def _extract_cards(self, html: str, base_url: Optional[str]) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        fragments = self._iter_anchor_fragments(html)
        if not fragments:
            fragments = self._iter_block_fragments(html)
        for attrs, body in fragments:
            title = self._extract_name(body)
            description = self._summarize_text(body)
            outer_link = self._extract_outer_link(attrs)
            link = self._resolve_link(outer_link or self._extract_link(body), base_url)
            founders = self._extract_founders(body) or []
            company_name = title or description[:60] or "listing"
            cards.append({
                "company_name": company_name,
                "name": company_name,
                "description": description,
                "source_url": link,
                "founders": founders,
            })
        return cards

    def _looks_like_card(self, attrs: str) -> bool:
        lowered = attrs.lower()
        return any(token in lowered for token in _CARD_CLASS_HINTS)

    def _extract_name(self, body: str) -> str:
        heading = _HEADING_PATTERN.search(body)
        if heading:
            return self._clean_html(heading.group("text"))
        alt = _NAME_PATTERN.search(body)
        if alt:
            return self._clean_html(alt.group("text"))
        return ""

    def _extract_link(self, body: str) -> Optional[str]:
        match = _LINK_PATTERN.search(body)
        if not match:
            return None
        return match.group("href")

    def _iter_anchor_fragments(self, html: str) -> List[tuple[str, str]]:
        fragments: List[tuple[str, str]] = []
        for match in _ANCHOR_CARD_PATTERN.finditer(html or ""):
            attrs = match.group("attrs") or ""
            if not self._looks_like_card(attrs):
                continue
            fragments.append((attrs, match.group("body") or ""))
        return fragments

    def _iter_block_fragments(self, html: str) -> List[tuple[str, str]]:
        fragments: List[tuple[str, str]] = []
        for match in _BLOCK_CARD_FRAGMENT.finditer(html or ""):
            attrs = match.group("attrs") or ""
            if not self._looks_like_card(attrs):
                continue
            fragments.append((attrs, match.group("body") or ""))
        return fragments

    def _extract_outer_link(self, attrs: str) -> Optional[str]:
        match = _ATTR_LINK_PATTERN.search(attrs or "")
        if not match:
            return None
        return match.group("href")

    def _resolve_link(self, link: Optional[str], base_url: Optional[str]) -> Optional[str]:
        if not link:
            return base_url
        if base_url:
            return urljoin(base_url, link)
        return link

    def _extract_founders(self, body: str) -> Optional[List[str]]:
        match = _FOUNDERS_PATTERN.search(body)
        if not match:
            return None
        names = [segment.strip().strip(",") for segment in match.group("names").split(" and ")]
        flattened: List[str] = []
        for name in names:
            flattened.extend(part.strip() for part in name.split(",") if part.strip())
        return [name for name in flattened if name]

    def _summarize_text(self, body: str) -> str:
        stripped = self._clean_html(body)
        return " ".join(stripped.split())[:280]

    def _clean_html(self, value: str) -> str:
        return re.sub(r"<[^>]+>", " ", value or "").strip()

    def _select_card(self, cards: List[Dict[str, Any]]) -> Dict[str, Any]:
        ranked = sorted(cards, key=lambda item: (item.get("name", "").lower(), item.get("description", "")))
        return ranked[0]

    def _normalize_card(self, card: Dict[str, Any], fallback_url: Optional[str]) -> Dict[str, Any]:
        normalized = dict(card)
        company_name = normalized.get("company_name") or normalized.get("name") or "listing"
        normalized["company_name"] = company_name
        normalized["name"] = company_name
        founders = normalized.get("founders") or []
        if not isinstance(founders, list):
            founders = [founders]
        normalized["founders"] = [name for name in founders if name]
        if not normalized.get("source_url"):
            normalized["source_url"] = fallback_url
        return normalized


__all__ = ["ListingExtractionSkill"]
