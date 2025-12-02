from __future__ import annotations

import pytest

from eikon_engine.utils.logging_utils import ArtifactLogger
from eikon_engine.workers.browser_worker import BrowserWorker


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:  # pragma: no cover - simple stub
        return None


@pytest.mark.asyncio
async def test_browser_worker_navigate_and_snapshot(tmp_path, monkeypatch):
    logger = ArtifactLogger(root=tmp_path, prefix="test")
    worker = BrowserWorker(settings={}, logger=logger, enable_playwright=False)

    async def fake_get(url: str):  # type: ignore[override]
        return _FakeResponse("<html><body>Secure Area</body></html>")

    monkeypatch.setattr(worker._http_client, "get", fake_get)

    result = await worker.execute({"action": {"action": "navigate", "url": "https://example.com"}})
    assert result["completion"]["complete"] is True
    assert "secure area" in (result["dom_snapshot"] or "").lower()
    assert result["layout_graph"].startswith("html")


@pytest.mark.asyncio
async def test_browser_worker_screenshot(tmp_path, monkeypatch):
    logger = ArtifactLogger(root=tmp_path, prefix="test")
    worker = BrowserWorker(settings={}, logger=logger, enable_playwright=False)

    async def fake_get(url: str):  # type: ignore[override]
        return _FakeResponse("<html><body>Example</body></html>")

    monkeypatch.setattr(worker._http_client, "get", fake_get)

    actions = [
        {"action": "navigate", "url": "https://example.com"},
        {"action": "screenshot", "name": "demo.png"},
    ]
    result = await worker.execute({"action": actions})
    assert any(path.endswith("demo.png") for path in result["screenshots"])
