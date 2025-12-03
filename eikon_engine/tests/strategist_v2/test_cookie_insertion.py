from __future__ import annotations

from eikon_engine.strategist.strategist_v2 import StrategistV2

COOKIE_HTML = """
<div class="cookie-banner">
  <p>We use cookies and similar technologies</p>
  <button class="accept">Accept Cookies</button>
</div>
"""


class StubPlanner:
    async def create_plan(self, goal: str, last_result=None):  # noqa: D401
        return make_plan()


def make_plan() -> dict:
    return {
        "plan_id": "demo",
        "goal": "demo",
        "tasks": [
            {
                "id": "task_1",
                "tool": "BrowserWorker",
                "bucket": "navigation",
                "depends_on": [],
                "inputs": {"actions": [{"id": "s1", "action": "navigate", "url": "https://site"}]},
            },
            {
                "id": "task_2",
                "tool": "BrowserWorker",
                "bucket": "interaction",
                "depends_on": ["task_1"],
                "inputs": {
                    "actions": [
                        {"id": "s2", "action": "click", "selector": "#cta"},
                    ]
                },
            },
        ],
    }


def make_strategist() -> StrategistV2:
    return StrategistV2(planner=StubPlanner())


def test_cookie_popup_inserts_step() -> None:
    strategist = make_strategist()
    strategist.load_plan(make_plan())
    run_ctx = {"current_url": "https://site"}
    first_step = strategist.next_step()
    result = {
        "completion": {"complete": True, "reason": "navigate"},
        "dom_snapshot": COOKIE_HTML,
        "meta": {"url": "https://site"},
    }
    strategist.on_step_result(run_ctx, first_step.metadata, result)
    next_step = strategist.peek_step()
    assert next_step["source"] == "cookie_popup"
    assert next_step["action_payload"]["action"] == "click"
    assert next_step["action_payload"].get("name") == "dismiss_cookie"


def test_cookie_popup_uses_dom_artifact(tmp_path) -> None:
    strategist = make_strategist()
    strategist.load_plan(make_plan())
    run_ctx = {"current_url": "https://site"}
    first_step = strategist.next_step()
    dom_path = tmp_path / "dom.html"
    dom_path.write_text(COOKIE_HTML, encoding="utf-8")
    result = {
        "completion": {"complete": False, "reason": "navigate"},
        "dom_snapshot": None,
        "failure_dom_path": str(dom_path),
        "meta": {"url": "https://site"},
    }
    strategist.on_step_result(run_ctx, first_step.metadata, result)
    next_step = strategist.peek_step()
    assert next_step["source"] == "cookie_popup"
