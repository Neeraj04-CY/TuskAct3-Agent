from __future__ import annotations

from pathlib import Path

from eikon_engine.stability import StabilityMonitor
from eikon_engine.strategist.agent_memory import AgentMemory


def test_stability_summary_persists_into_agent_memory(tmp_path: Path) -> None:
    monitor = StabilityMonitor(history_path=tmp_path / "history.json")
    run_ctx = {
        "reward_trace": [
            {"step_id": "s-last", "reward": 0.8, "confidence": {"confidence": 0.75}},
        ],
        "repair_events": [],
        "history": [],
        "current_fingerprint": "memory_fingerprint",
    }

    report = monitor.evaluate_run(
        goal="integration",
        completion={"complete": True, "reason": "ok"},
        run_context=run_ctx,
        strategist_trace=[],
        duration_seconds=4.0,
        artifact_base=None,
    )

    memory = AgentMemory()
    memory.store_stability("memory_fingerprint", report["metrics"])

    hint = memory.retrieve("memory_fingerprint")
    assert hint is not None
    assert hint["stability"]["avg_reward"] == report["metrics"]["avg_reward"]
    assert hint["stability"]["success_rate"] == report["metrics"]["success_rate"]
