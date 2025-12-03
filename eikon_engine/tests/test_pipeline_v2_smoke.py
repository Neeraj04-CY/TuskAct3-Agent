from __future__ import annotations

from pathlib import Path

from eikon_engine.pipelines.browser_pipeline import run_pipeline


def test_pipeline_v2_smoke(tmp_path) -> None:  # noqa: ARG001
    demo_html = Path("examples/demo_local_testsite/login.html").resolve()
    assert demo_html.exists(), "demo HTML fixture is missing"
    goal = f"Open {demo_html.as_uri()} and log in using demo credentials"

    result = run_pipeline(goal, dry_run=True, allow_sensitive=False)

    assert isinstance(result, dict)
    assert "completion" in result

    strategist_keys = ("strategist_decisions", "strategy_events", "strategist_trace")
    assert any(key in result for key in strategist_keys), "Strategist metadata not found in pipeline result"
