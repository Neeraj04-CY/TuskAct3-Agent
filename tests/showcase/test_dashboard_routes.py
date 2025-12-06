from __future__ import annotations

import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient


def _seed_run(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts" / "autonomy"
    run_dir = artifacts / "run_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    screenshot = run_dir / "shot.png"
    screenshot.write_bytes(b"fakeimage")

    summary = {
        "goal": "test",
        "completed": True,
        "reason": "ok",
        "duration_seconds": 12,
        "step_count": 4,
        "reward_trace": [
            {"step_id": "s1", "reward": 0.2, "confidence": {"confidence": 0.6}},
            {"step_id": "s2", "reward": 0.8, "confidence": {"confidence": 0.9}},
        ],
        "repair_events": [{"patch": {"type": "selector"}}],
    }
    result = {
        "run_context": {
            "reward_trace": summary["reward_trace"],
            "repair_events": summary["repair_events"],
            "plan_evolution": [{"cursor": 1, "needs_replan": False}],
            "behavior_predictions": [{"step_id": "s1", "difficulty": 0.4, "selector_bias": "css", "likely_repair": False}],
            "memory_summary": {"entries": 3, "avg_confidence": 0.8, "avg_difficulty": 0.5},
        },
        "steps": [
            {
                "step": {"step_id": "s1", "action": "click"},
                "result": {"dom_snapshot": "<div>ok</div>", "screenshot_path": str(screenshot)},
            }
        ],
    }
    stability = {
        "metrics": {
            "avg_reward": 0.5,
            "avg_confidence": 0.7,
            "repeated_failures": {"timeout": 2},
        },
        "history_snapshot": [
            {"timestamp": "2025-12-01", "metrics": {"avg_reward": 0.4, "avg_confidence": 0.6}},
            {"timestamp": "2025-12-02", "metrics": {"avg_reward": 0.5, "avg_confidence": 0.7}},
        ],
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (run_dir / "stability_report.json").write_text(json.dumps(stability, indent=2), encoding="utf-8")
    latest = {"run_path": str(run_dir), "summary_path": str(run_dir / "summary.json"), "timestamp": "test"}
    (artifacts / "latest_run.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")


def test_dashboard_route_returns_payload(monkeypatch, tmp_path: Path) -> None:
    _seed_run(tmp_path)
    monkeypatch.setenv("EIKON_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    from dashboard import server

    importlib.reload(server)

    client = TestClient(server.app)
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["goal"] == "test"
    assert payload["dom_assets"][0]["step_id"] == "s1"
    assert payload["repeated_failures"][0]["reason"] == "timeout"
