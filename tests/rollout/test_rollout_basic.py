from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from eikon_engine.stability import StabilityMonitor
from run_rollout import EpisodeRequest, RolloutEngine


def _make_payload(index: int, completed: bool = True) -> Dict[str, Dict[str, object]]:
    reward_trace: List[Dict[str, object]] = [
        {"step_id": f"s{index}", "reward": float(index), "confidence": {"confidence": 0.5 + 0.05 * index}}
    ]
    run_ctx = {
        "reward_trace": reward_trace,
        "repair_events": [{"patch": {"type": "selector"}} for _ in range(index)],
        "plan_evolution": [{"cursor": index, "needs_replan": False}],
        "behavior_predictions": [
            {
                "difficulty": 0.2 + 0.1 * index,
                "likely_repair": index % 2 == 0,
                "fingerprint": f"fp_{index}",
            }
        ],
        "memory_summary": {"entries": index, "avg_confidence": 0.4 + 0.1 * index, "avg_difficulty": 0.3 + 0.05 * index},
        "current_fingerprint": f"fp_{index}",
    }
    stability = {
        "timestamp": "2025-12-03T00:00:00Z",
        "goal": "demo",
        "metrics": {
            "avg_reward": float(index),
            "avg_confidence": 0.5,
            "repair_count": index,
            "duration_seconds": 4.0 + index,
            "dom_fingerprint": f"fp_{index}",
            "reward_drift": -0.1 * index,
            "repeated_failures": {"timeout": index - 1} if index > 1 else {},
        },
        "trends": {},
    }
    result = {
        "goal": "demo rollout",
        "completion": {"complete": completed, "reason": "ok" if completed else "timeout"},
        "run_context": run_ctx,
        "duration_seconds": 5.0 + index,
        "stability": stability,
    }
    summary = {"goal": "demo rollout", "completed": completed, "reason": "ok"}
    return {"result": result, "summary": summary}


def test_rollout_basic(tmp_path: Path) -> None:
    payloads = {
        1: _make_payload(1, completed=True),
        2: _make_payload(2, completed=False),
    }

    def runner(request: EpisodeRequest) -> Dict[str, Dict[str, object]]:
        return payloads[request.index]

    monitor = StabilityMonitor(history_path=tmp_path / "history.json")
    engine = RolloutEngine(
        goal="demo rollout",
        episodes=2,
        output_root=tmp_path / "rollouts",
        stability_monitor=monitor,
        episode_runner=runner,
    )
    summary = engine.run()

    assert summary["episodes"] == 2
    run_one = tmp_path / "rollouts" / "run_01"
    assert (run_one / "signals.json").exists()
    assert (run_one / "stability_report.json").exists()
    with (run_one / "signals.json").open("r", encoding="utf-8") as handle:
        data = json.load(handle)
        assert data["reward_trace"]
        assert data["planner_evolution"]
    assert (tmp_path / "rollouts" / "rollout_summary.json").exists()
    assert (tmp_path / "rollouts" / "rollout_summary.md").exists()
