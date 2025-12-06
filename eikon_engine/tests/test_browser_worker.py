from __future__ import annotations

from pathlib import Path

import pytest

from eikon_engine.utils.logging_utils import ArtifactLogger
from eikon_engine.workers.browser_worker import BrowserWorker


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:  # pragma: no cover - simple stub
        return None


HEROKU_LOGIN_HTML = """
<html>
    <body>
        <div id="content">
            <form id="login" action="/authenticate">
                <div class="field">
                    <label for="username">Username</label>
                    <input id="username" name="username" type="text" />
                </div>
                <div class="field">
                    <label for="password">Password</label>
                    <input id="password" name="password" type="password" />
                </div>
                <button class="radius" type="submit">Login</button>
            </form>
        </div>
    </body>
</html>
"""


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


@pytest.mark.asyncio
async def test_browser_worker_failure_artifacts(tmp_path, monkeypatch):
    logger = ArtifactLogger(root=tmp_path, prefix="test")
    worker = BrowserWorker(settings={}, logger=logger, enable_playwright=False)

    async def fake_get(url: str):  # type: ignore[override]
        raise RuntimeError("boom")

    monkeypatch.setattr(worker._http_client, "get", fake_get)

    result = await worker.execute({"action": {"action": "navigate", "url": "https://example.com"}})
    assert result["error"] is not None
    assert result["failure_dom_path"] is not None
    assert Path(result["failure_dom_path"]).exists()
    assert result["failure_screenshot_path"] is not None
    assert Path(result["failure_screenshot_path"]).exists()


@pytest.mark.asyncio
async def test_dom_presence_uses_login_overrides_for_heroku(tmp_path, monkeypatch):
    logger = ArtifactLogger(root=tmp_path, prefix="test")
    worker = BrowserWorker(settings={}, logger=logger, enable_playwright=False)
    worker.set_mission_context(mission_instruction="Login demo: the-internet.herokuapp.com/login")

    async def fake_get(url: str):  # type: ignore[override]
        return _FakeResponse(HEROKU_LOGIN_HTML)

    monkeypatch.setattr(worker._http_client, "get", fake_get)

    actions = [
        {"action": "navigate", "url": "https://the-internet.herokuapp.com/login"},
        {"action": "dom_presence_check", "selector": "input"},
    ]
    result = await worker.execute({"action": actions, "goal": "Login demo"})

    dom_presence = result.get("dom_presence") or []
    assert dom_presence, "expected dom_presence events"
    check = dom_presence[0]
    assert check["status"] == "ok"
    assert check["selectors"][0] == "#username"
    resolver_meta = check.get("selector_resolver") or {}
    assert resolver_meta.get("context", {}).get("heroku_override") is True
