"""Minimal DOM helpers for snapshotting and layout graph generation."""

from __future__ import annotations

from html.parser import HTMLParser
from typing import List


class _TagCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        self.tags.append(tag)


def build_layout_graph(html: str) -> str:
    """Return a simple textual graph summary from HTML."""

    parser = _TagCollector()
    parser.feed(html or "")
    unique_tags = list(dict.fromkeys(parser.tags))
    return " -> ".join(unique_tags[:25]) or "document"
