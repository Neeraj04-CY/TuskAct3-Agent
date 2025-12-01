from __future__ import annotations

from src.tool_policy import ToolPolicy


def test_tool_policy_rejects_dangerous_command() -> None:
    policy = ToolPolicy(allowed_workers={"BrowserWorker"})
    decision = policy.evaluate({"worker": "BrowserWorker", "description": "Delete entire filesystem"})
    assert decision.allowed is False
    assert "Dangerous" in (decision.reason or "")


def test_tool_policy_allows_safe_worker() -> None:
    policy = ToolPolicy(allowed_workers={"BrowserWorker", "Writer"})
    decision = policy.evaluate({"worker": "Writer", "description": "Summarize article"})
    assert decision.allowed is True
    assert decision.reason is None
