from __future__ import annotations

from pathlib import Path

from eikon_engine.stability import StabilityMonitor


def _sample_run_context() -> dict:
    return {
        "reward_trace": [
            {"step_id": "s1", "reward": 1.0, "confidence": {"confidence": 0.85, "band": "high"}},
            {"step_id": "s2", "reward": 0.6, "confidence": {"confidence": 0.7, "band": "medium"}},
        ],
        "repair_events": [
            {"patch": {"type": "selector"}},
        ],
        "history": [
            {"status": "ok"},
            {"status": "timeout"},
        ],
        "current_fingerprint": "abc123",
    }


def test_stability_monitor_generates_reports(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    monitor = StabilityMonitor(history_path=history_path)

    report = monitor.evaluate_run(
        goal="demo",
        completion={"complete": True, "reason": "ok"},
        run_context=_sample_run_context(),
        strategist_trace=[],
        duration_seconds=6.5,
        artifact_base="runs/sample",
    )

    assert report["metrics"]["success_rate"] == 1.0
    assert report["metrics"]["avg_reward"] > 0

    paths = monitor.write_reports(report, tmp_path)
    assert paths["json"].exists()
    assert "Stability Report" in paths["markdown"].read_text(encoding="utf-8")
    assert history_path.exists()
