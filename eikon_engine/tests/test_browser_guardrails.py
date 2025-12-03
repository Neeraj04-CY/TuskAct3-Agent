from __future__ import annotations

import pytest

from eikon_engine.workers.browser_worker import BrowserWorker


@pytest.mark.asyncio
async def test_guardrails_block_sensitive_screenshot() -> None:
    worker = BrowserWorker(settings={"guardrails": {}})
    result = await worker.execute({"action": {"action": "screenshot", "name": "password_field"}})
    await worker.close()
    step = result["steps"][0]
    assert step["status"] == "blocked"
    assert step["block_reason"] == "screenshot_blocked_sensitive"


@pytest.mark.asyncio
async def test_guardrails_block_risky_click() -> None:
    worker = BrowserWorker(settings={"guardrails": {}})
    result = await worker.execute({"action": {"action": "click", "selector": "#delete-account"}})
    await worker.close()
    step = result["steps"][0]
    assert step["status"] == "blocked"
    assert step["block_reason"] == "click_blocked_risky"