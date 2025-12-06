from __future__ import annotations

from pathlib import Path
from typing import Dict

from eikon_engine.stability import StabilityMonitor
from run_rollout import EpisodeRequest, RolloutEngine

from .test_rollout_basic import _make_payload


def test_rollout_integration_outputs_memory_and_success(tmp_path: Path) -> None:
    payloads = {
        1: _make_payload(1, completed=True),
        2: _make_payload(2, completed=True),
        3: _make_payload(3, completed=False),
    }

    def runner(request: EpisodeRequest) -> Dict[str, Dict[str, object]]:
        return payloads[request.index]

    monitor = StabilityMonitor(history_path=tmp_path / "history.json")
    engine = RolloutEngine(
        goal="integration goal",
        episodes=3,
        output_root=tmp_path / "rollouts",
        stability_monitor=monitor,
        episode_runner=runner,
    )
    summary = engine.run()

    assert len(summary["success_classification"]) == 3
    assert summary["successes"] == 2
    assert summary["stability_drift"]["scores"]
    per_run = summary["per_run"]
    assert per_run[0]["avg_reward"] < per_run[-1]["avg_reward"]
