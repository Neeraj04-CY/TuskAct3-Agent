from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi.testclient import TestClient


def _seed_artifacts(root: Path) -> None:
    run_dir = root / "autonomy" / "run_seed"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {"goal": "cached", "completed": True, "reason": "ok", "reward_trace": []}
    result = {"run_context": {"reward_trace": [], "behavior_predictions": [], "memory_summary": {}}, "steps": []}
    stability = {"metrics": {"avg_reward": 0.5, "avg_confidence": 0.7, "repeated_failures": {}}, "history_snapshot": []}
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (run_dir / "stability_report.json").write_text(json.dumps(stability), encoding="utf-8")
    latest = {"run_path": str(run_dir), "summary_path": str(run_dir / "summary.json"), "timestamp": "seed"}
    (root / "autonomy" / "latest_run.json").write_text(json.dumps(latest), encoding="utf-8")
    asset = run_dir / "asset.txt"
    asset.write_text("artifact", encoding="utf-8")


def test_api_server_endpoints(monkeypatch, tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _seed_artifacts(artifact_root)
    monkeypatch.setenv("EIKON_ARTIFACT_ROOT", str(artifact_root))

    import api_server

    importlib.reload(api_server)

    def fake_run(goal: str, **_: Any) -> Dict[str, Any]:
        return {
            "summary": {"goal": goal, "completed": True, "reason": "ok"},
            "run_dir": str(artifact_root / "autonomy" / "run_seed"),
            "stability": {"metrics": {}},
        }

    class _FakePlanner:
        async def create_plan(self, goal: str, *, last_result: Dict[str, Any] | None = None) -> Dict[str, Any]:
            return {"goal": goal, "tasks": []}

    class _FakeLearner:
        def predict(self, fingerprint: str, rewards: List[float], repairs: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {"fingerprint": fingerprint, "difficulty": 0.5, "likely_repair": False, "selector_bias": "css", "recommended_subgoals": []}

    api_server.run_single_demo = fake_run  # type: ignore
    api_server._planner_adapter = _FakePlanner()  # type: ignore
    api_server._behavior_learner = _FakeLearner()  # type: ignore

    client = TestClient(api_server.app)

    run_resp = client.post("/run", json={"goal": "api demo"})
    assert run_resp.status_code == 200
    assert run_resp.json()["summary"]["goal"] == "api demo"

    plan_resp = client.post("/plan", json={"goal": "api demo"})
    assert plan_resp.status_code == 200
    assert plan_resp.json()["plan"]["goal"] == "api demo"

    predict_resp = client.post("/predict", json={"fingerprint": "fp", "recent_rewards": [0.2], "repair_events": []})
    assert predict_resp.status_code == 200
    assert predict_resp.json()["selector_bias"] == "css"

    last_resp = client.get("/last_run")
    assert last_resp.status_code == 200
    assert last_resp.json()["timestamp"] == "seed"

    artifact_resp = client.get("/artifacts/autonomy/run_seed/asset.txt")
    assert artifact_resp.status_code == 200
    assert artifact_resp.text == "artifact"

    dashboard_resp = client.get("/dashboard")
    assert dashboard_resp.status_code == 200
    assert dashboard_resp.json()["available"] is True