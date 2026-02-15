from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser
from typing import Any, Dict, List

CARD_KEYWORDS = ("card", "listing", "result", "company", "startup", "item", "tile")
NAV_KEYWORDS = ("nav", "sidebar", "menu", "dashboard")
HEADING_TAGS = {"h1", "h2", "h3", "h4"}
LIST_TAGS = {"ul", "ol"}
CARD_TAGS = {"div", "article", "section", "li"}


class _DomSignalParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tag_counts: Counter[str] = Counter()
        self.input_types: Counter[str] = Counter()
        self.class_counts: Counter[str] = Counter()
        self.card_hits: Counter[str] = Counter()
        self.text_chars = 0
        self.link_count = 0
        self.list_item_count = 0
        self.list_link_hits = 0
        self.heading_count = 0
        self.paragraph_count = 0
        self.nav_score = 0
        self.stack: List[str] = []
        self._in_list = False

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        self.stack.append(tag_lower)
        self.tag_counts[tag_lower] += 1
        attr_map = {key: (value or "") for key, value in attrs}
        class_value = attr_map.get("class", "")
        if class_value:
            for token in class_value.split():
                normalized = token.strip().lower()
                if not normalized:
                    continue
                self.class_counts[normalized] += 1
                if any(keyword in normalized for keyword in CARD_KEYWORDS):
                    self.card_hits[normalized] += 1
                if any(keyword in normalized for keyword in NAV_KEYWORDS):
                    self.nav_score += 1
        if tag_lower in LIST_TAGS:
            self._in_list = True
        if tag_lower == "li":
            self.list_item_count += 1
        if tag_lower == "a":
            self.link_count += 1
            if self._in_list:
                self.list_link_hits += 1
        if tag_lower in HEADING_TAGS:
            self.heading_count += 1
        if tag_lower == "p":
            self.paragraph_count += 1
        if tag_lower == "input":
            input_type = (attr_map.get("type") or "text").lower()
            self.input_types[input_type] += 1
        if tag_lower == "nav" or attr_map.get("role", "").lower() == "navigation":
            self.nav_score += 1

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in LIST_TAGS:
            self._in_list = False
        if self.stack:
            self.stack.pop()

    def handle_data(self, data: str) -> None:
        self.text_chars += len(data.strip())

    def summary(self) -> Dict[str, Any]:
        card_repetition = max(self.card_hits.values(), default=0)
        list_link_ratio = 0.0
        if self.list_item_count:
            list_link_ratio = round(self.list_link_hits / self.list_item_count, 3)
        return {
            "form_count": self.tag_counts.get("form", 0),
            "password_inputs": self.input_types.get("password", 0),
            "input_count": self.tag_counts.get("input", 0),
            "card_repetition": card_repetition,
            "card_class_max": max(self.card_hits, key=self.card_hits.get, default=""),
            "list_link_ratio": list_link_ratio,
            "list_item_count": self.list_item_count,
            "link_count": self.link_count,
            "article_tags": self.tag_counts.get("article", 0),
            "table_count": self.tag_counts.get("table", 0),
            "nav_score": self.nav_score,
            "heading_count": self.heading_count,
            "paragraph_count": self.paragraph_count,
            "text_density": round(self.text_chars / max(self.tag_counts.total() or 1, 1), 3),
        }


def extract_page_signals(dom: str) -> Dict[str, Any]:
    parser = _DomSignalParser()
    parser.feed(dom or "")
    parser.close()
    summary = parser.summary()
    summary["char_count"] = parser.text_chars
    return summary


__all__ = ["extract_page_signals"]
