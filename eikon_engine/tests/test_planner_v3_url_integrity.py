from __future__ import annotations

from eikon_engine.planning.planner_v3 import extract_known_urls


def test_url_integrity() -> None:
    goal = "Open file:///C:/path/to/demo/login.html?next=%2Fdashboard and then go to https://example.com/app"
    urls = extract_known_urls(goal)
    assert "file:///C:/path/to/demo/login.html?next=%2Fdashboard" in urls
    assert "https://example.com/app" in urls
    for url in urls:
        assert url.endswith("html?next=%2Fdashboard") or "example.com/app" in url
