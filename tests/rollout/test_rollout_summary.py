from __future__ import annotations

from pathlib import Path
from typing import Dict

from eikon_engine.stability import StabilityMonitor
from run_rollout import EpisodeRequest, RolloutEngine

from .test_rollout_basic import _make_payload


def test_rollout_summary_trends(tmp_path: Path) -> None:
    payloads = {idx: _make_payload(idx, completed=True) for idx in range(1, 4)}

    def runner(request: EpisodeRequest) -> Dict[str, Dict[str, object]]:
        return payloads[request.index]

    monitor = StabilityMonitor(history_path=tmp_path / "history.json")
    engine = RolloutEngine(
        goal="trend goal",
        episodes=3,
        output_root=tmp_path / "rollouts",
        stability_monitor=monitor,
        episode_runner=runner,
    )
    summary = engine.run()

    assert summary["reward_trend"]["slope"] > 0
    assert summary["repair_trend"]["slope"] > 0
    clusters = summary["repeated_failure_clusters"]
    assert clusters and clusters[0]["reason"] == "timeout"
    # Memory growth should reflect more entries by the final run
    memory_growth = summary["memory_growth"]
    assert memory_growth["entries_end"] >= memory_growth["entries_start"]
