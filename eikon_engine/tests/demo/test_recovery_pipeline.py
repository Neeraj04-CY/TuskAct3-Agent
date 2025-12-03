from __future__ import annotations

import asyncio
import copy

from eikon_engine.core.completion import build_completion
from eikon_engine.core.orchestrator_v2 import OrchestratorV2
from eikon_engine.strategist.strategist_v2 import StrategistV2

LOGIN_URL = "https://demo/login"
DASHBOARD_URL = "https://demo/dashboard"

COOKIE_DOM = """
<div>
  <div class="cookie-banner">
    <p>We use cookies to personalize content.</p>
    <button class="accept">Accept All</button>
  </div>
  <form>
        <input id="email-input" name="email" value="" />
        <input id="password-input" type="password" value="" />
    <button id="loginButton">Log In</button>
  </form>
</div>
"""

LOGIN_DOM = """
<div>
  <form>
        <input id="email-input" name="email" value="" />
        <input id="password-input" type="password" value="" />
    <button id="loginButton">Log In</button>
  </form>
</div>
"""

MUTATED_BUTTON_DOM = """
<div>
    <form>
        <input id="email-input" name="email" value="" />
        <input id="password-input" type="password" value="" />
        <button id="primary-login-button" data-testid="primary-login-button">Log In</button>
    </form>
</div>
"""

DASHBOARD_DOM = """
<div class="app-shell">
  <h1>Team Dashboard</h1>
  <section class="cards">
    <button class="card">View Usage</button>
    <button class="card">Invoices</button>
  </section>
</div>
"""


def make_login_plan(actions: list[dict[str, str]] | None = None) -> dict:
    plan_actions = actions or [
        {"id": "s1", "action": "navigate", "url": LOGIN_URL},
        {"id": "s2", "action": "fill", "selector": "#email-input", "value": "demo@example.com"},
        {"id": "s3", "action": "fill", "selector": "#password-input", "value": "hunter2"},
        {
            "id": "s4",
            "action": "click",
            "selector": "#loginButton",
            "name": "submit_login",
            "text": "Log In",
        },
    ]
    return {
        "plan_id": "demo",
        "goal": "demo",
        "tasks": [
            {
                "id": "task_login",
                "tool": "BrowserWorker",
                "bucket": "auth",
                "depends_on": [],
                "inputs": {"actions": plan_actions},
            }
        ],
    }


class StaticPlanner:
    def __init__(self, plan: dict) -> None:
        self.plan = plan

    async def create_plan(self, goal: str, *, last_result=None):  # noqa: D401
        _ = goal, last_result
        return copy.deepcopy(self.plan)


class ScriptedWorker:
    def __init__(self, script: list[dict]) -> None:
        self._script = list(script)
        self.calls: list[dict] = []

    async def execute(self, metadata: dict) -> dict:
        if not self._script:
            raise AssertionError("Script exhausted")
        entry = self._script.pop(0)
        action = metadata.get("action", {})
        expect = entry.get("expect", {})
        if "action" in expect:
            assert action.get("action") == expect["action"], f"expected action {expect['action']} got {action}"
        if "selector" in expect:
            assert action.get("selector") == expect["selector"], f"selector mismatch for action {action}"
        if "name" in expect:
            assert action.get("name") == expect["name"], f"name mismatch for action {action}"
        self.calls.append(action)
        result = copy.deepcopy(entry["result"])
        result.setdefault("steps", [action])
        result.setdefault("meta", {})
        return result

    async def close(self) -> None:  # pragma: no cover - simple stub
        return None


def worker_result(
    *,
    dom: str,
    meta: dict | None = None,
    complete: bool = True,
    reason: str = "ok",
    error: str | None = None,
) -> dict:
    return {
        "steps": [],
        "screenshots": [],
        "dom_snapshot": dom,
        "layout_graph": "<graph />",
        "completion": build_completion(complete=complete and error is None, reason=reason, payload={}),
        "error": error,
        "meta": meta or {},
        "failure_dom_path": None,
        "failure_screenshot_path": None,
    }


def test_cookie_banner_recovery_in_pipeline() -> None:
    strategist = StrategistV2(planner=StaticPlanner(make_login_plan()))
    script = [
        {"expect": {"action": "navigate"}, "result": worker_result(dom=COOKIE_DOM, meta={"url": LOGIN_URL})},
        {
            "expect": {"action": "click", "name": "dismiss_cookie"},
            "result": worker_result(dom=LOGIN_DOM, meta={"url": LOGIN_URL}),
        },
        {
            "expect": {"action": "fill", "selector": "#email-input"},
            "result": worker_result(dom=LOGIN_DOM, meta={"url": LOGIN_URL}),
        },
        {
            "expect": {"action": "fill", "selector": "#password-input"},
            "result": worker_result(dom=LOGIN_DOM, meta={"url": LOGIN_URL}),
        },
        {
            "expect": {"action": "click", "selector": "#loginButton"},
            "result": worker_result(dom=DASHBOARD_DOM, meta={"url": DASHBOARD_URL}),
        },
    ]
    worker = ScriptedWorker(script)
    orchestrator = OrchestratorV2(strategist=strategist, worker=worker, max_steps=10)

    result = asyncio.run(orchestrator.run_goal("Demo login with cookies"))

    assert result["completion"]["complete"] is True
    assert any(entry["step"].get("source") == "cookie_popup" for entry in result["steps"])
    assert result["steps"][1]["step"].get("source") == "cookie_popup"


def test_selector_repair_flow_in_pipeline() -> None:
    strategist = StrategistV2(planner=StaticPlanner(make_login_plan()))
    script = [
        {"expect": {"action": "navigate"}, "result": worker_result(dom=LOGIN_DOM, meta={"url": LOGIN_URL})},
        {
            "expect": {"action": "fill", "selector": "#email-input"},
            "result": worker_result(dom=LOGIN_DOM, meta={"url": LOGIN_URL}),
        },
        {
            "expect": {"action": "fill", "selector": "#password-input"},
            "result": worker_result(dom=LOGIN_DOM, meta={"url": LOGIN_URL}),
        },
        {
            "expect": {"action": "click", "selector": "#loginButton"},
            "result": worker_result(dom=MUTATED_BUTTON_DOM, meta={"url": LOGIN_URL}),
        },
        {
            "expect": {"action": "click", "selector": "#primary-login-button"},
            "result": worker_result(dom=DASHBOARD_DOM, meta={"url": DASHBOARD_URL}),
        },
    ]
    worker = ScriptedWorker(script)
    orchestrator = OrchestratorV2(strategist=strategist, worker=worker, max_steps=10)

    result = asyncio.run(orchestrator.run_goal("Demo login with selector repair"))

    repair_steps = [entry for entry in result["steps"] if entry["step"].get("source") == "micro_repair"]
    assert repair_steps, "micro repair step missing"
    assert repair_steps[0]["step"]["action_payload"]["selector"] == "#primary-login-button"
    assert result["completion"]["complete"] is True


def test_redirect_triggers_pipeline_replan() -> None:
    planner = StaticPlanner(make_login_plan(actions=[{"id": "s1", "action": "navigate", "url": LOGIN_URL}]))
    strategist = StrategistV2(planner=planner)
    script = [
        {
            "expect": {"action": "navigate"},
            "result": worker_result(
                dom=LOGIN_DOM,
                meta={"url": DASHBOARD_URL, "redirect_url": DASHBOARD_URL},
            ),
        }
    ]
    worker = ScriptedWorker(script)
    orchestrator = OrchestratorV2(strategist=strategist, worker=worker, max_steps=5)

    result = asyncio.run(orchestrator.run_goal("Demo redirect handling"))

    assert result["completion"]["complete"] is False
    assert result["completion"]["reason"] == "replan requested"
    redirects = result["run_context"].get("redirects") or []
    assert redirects and redirects[0]["to"] == DASHBOARD_URL