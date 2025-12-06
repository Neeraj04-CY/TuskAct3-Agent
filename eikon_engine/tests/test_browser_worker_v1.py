from __future__ import annotations

from pathlib import Path

import pytest

from eikon_engine.browser.worker_v1 import BrowserWorkerV1
from eikon_engine.utils.logging_utils import ArtifactLogger


@pytest.mark.asyncio
async def test_browser_worker_v1_runs_plan_success(tmp_path: Path) -> None:
    page = tmp_path / "page.html"
    page.write_text(
        """
        <html><body>
            <form>
                <input id="username" />
                <input id="password" />
                <button type="submit" id="login">Go</button>
            </form>
            <div class="flash success">Hi</div>
        </body></html>
        """,
        encoding="utf-8",
    )
    plan = {
        "plan_id": "plan-success",
        "goal": "Login to local page",
        "tasks": [
            {
                "id": "task_1",
                "tool": "BrowserWorker",
                "inputs": {
                    "actions": [
                        {"action": "navigate", "url": page.as_uri()},
                        {
                            "action": "fill",
                            "fields": [
                                {"selector": "#username", "value": "demo"},
                                {"selector": "#password", "value": "pass"},
                            ],
                        },
                        {"action": "click", "selector": "#login"},
                        {"action": "screenshot", "name": "login.png"},
                        {"action": "reload_if_failed", "_recovery": True},
                        {"action": "extract_dom"},
                    ]
                },
                "depends_on": [],
                "bucket": "navigation",
            }
        ],
    }

    logger = ArtifactLogger(base_dir=tmp_path / "logs")
    worker = BrowserWorkerV1(
        settings={"logging": {"artifact_root": str(tmp_path / "fallback_artifacts")}},
        logger=logger,
        enable_playwright=False,
    )

    summary = await worker.run_plan(plan)
    await worker.close()

    assert summary["plan_id"] == "plan-success"
    assert summary["failures"] == 0
    assert summary["recovery_steps"] == 1
    assert summary["final_url"].startswith("file:")
    assert len(summary["traces"]) == 6
    screenshot_trace = next(trace for trace in summary["traces"] if trace["action"] == "screenshot")
    assert screenshot_trace["screenshot_path"].endswith("login.png")
    assert Path(screenshot_trace["screenshot_path"]).exists()
    assert logger.summary_file.exists()


@pytest.mark.asyncio
async def test_browser_worker_v1_records_failure_on_dom_check(tmp_path: Path) -> None:
    page = tmp_path / "page.html"
    page.write_text("<html><body><div class='present'>Here</div></body></html>", encoding="utf-8")
    plan = {
        "plan_id": "plan-failure",
        "goal": "Check DOM",
        "tasks": [
            {
                "id": "task_1",
                "tool": "BrowserWorker",
                "inputs": {
                    "actions": [
                        {"action": "navigate", "url": page.as_uri()},
                        {"action": "dom_presence_check", "selector": ".missing"},
                        {"action": "screenshot", "name": "after.png"},
                    ]
                },
                "depends_on": [],
                "bucket": "navigation",
            }
        ],
    }

    logger = ArtifactLogger(base_dir=tmp_path / "logs")
    worker = BrowserWorkerV1(logger=logger, enable_playwright=False)

    summary = await worker.run_plan(plan)
    await worker.close()

    assert summary["failures"] == 1
    assert summary["total_steps"] == 2
    assert summary["first_failure_type"]
    assert "dom_presence_check" in summary["first_failure_type"]
    assert summary["recovery_steps"] == 0
    assert summary["run_output"].startswith("failed")
    error_trace = summary["traces"][-1]
    assert error_trace["status"] == "error"
    assert "dom_presence_failed" in (error_trace["error"] or "")
    assert error_trace["details"]["selectors"][0] == ".missing"
    assert not any(trace["action"] == "screenshot" for trace in summary["traces"] if trace["status"] == "ok")
    assert logger.summary_file.exists()
