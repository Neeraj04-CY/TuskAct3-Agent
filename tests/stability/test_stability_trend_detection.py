from __future__ import annotations

import json
from pathlib import Path

from eikon_engine.stability import StabilityMonitor


def test_trend_detection_uses_history(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    history_path.write_text(
        json.dumps(
            [
                {
                    "timestamp": "2025-12-02T00:00:00Z",
                    "goal": "demo",
                    "completed": True,
                    "metrics": {
                        "avg_reward": 1.0,
                        "avg_confidence": 0.9,
                        "repair_count": 1,
                        "duration_seconds": 8.0,
                        "dom_fingerprint": "fingerprint_a",
                    },
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    monitor = StabilityMonitor(history_path=history_path)

    run_ctx = {
        "reward_trace": [
            {"step_id": "s1", "reward": 0.4, "confidence": {"confidence": 0.45}},
        ],
        "repair_events": [{}, {}],
        "history": [{"status": "timeout"}, {"status": "timeout"}],
        "current_fingerprint": "fingerprint_b",
    }

    report = monitor.evaluate_run(
        goal="demo",
        completion={"complete": False, "reason": "timeout"},
        run_context=run_ctx,
        strategist_trace=[{"event": "failure", "signature": "timeout"}],
        duration_seconds=12.0,
        artifact_base=None,
    )

    assert report["metrics"]["reward_drift"] == -0.6
    assert report["metrics"]["repair_count"] == 2
    repeated = report["metrics"]["repeated_failures"]
    assert repeated["timeout"] >= 2
    assert report["trends"]["dom_similarity"] < 1.0
