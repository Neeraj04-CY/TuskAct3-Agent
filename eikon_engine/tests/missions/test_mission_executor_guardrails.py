from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from eikon_engine.missions.mission_executor import MissionExecutor
from eikon_engine.missions.mission_schema import MissionSpec, MissionSubgoal


@pytest.mark.asyncio
async def test_mission_executor_halts_when_budget_exceeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = MissionSpec(instruction="Respect budget", execute=False, autonomy_budget={"max_steps": 0})
    subgoal = MissionSubgoal(id="sg_budget", description="Do work", planner_metadata={})
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda _: [subgoal])

    async def fake_pipeline(self, **_: Dict[str, object]) -> Dict[str, object]:  # noqa: ARG001
        return {
            "completion": {"complete": True, "reason": "ok"},
            "artifacts": {},
            "steps": [{"action": "navigate"}],
        }

    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", fake_pipeline)
    executor = MissionExecutor(
        settings={"planner": {}, "logging": {"artifact_root": str(tmp_path)}},
        artifacts_root=tmp_path,
    )
    result = await executor.run_mission(spec)

    assert result.status == "halted"
    assert result.termination["reason"] == "autonomy_budget"
    assert result.summary["termination"]["state"] == "HALTED"
    assert result.summary["autonomy_budget"]["steps_used"] >= 1


@pytest.mark.asyncio
async def test_mission_executor_enforces_safety_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = MissionSpec(
        instruction="Block download",
        execute=False,
        safety_contract={"blocked_actions": ["download_file"]},
    )
    subgoal = MissionSubgoal(id="sg_safe", description="Download file", planner_metadata={})
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda _: [subgoal])

    async def fake_pipeline(self, **_: Dict[str, object]) -> Dict[str, object]:  # noqa: ARG001
        return {
            "completion": {"complete": True, "reason": "ok"},
            "artifacts": {},
            "steps": [{"action": "download_file"}],
        }

    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", fake_pipeline)
    executor = MissionExecutor(
        settings={"planner": {}, "logging": {"artifact_root": str(tmp_path)}},
        artifacts_root=tmp_path,
    )
    result = await executor.run_mission(spec)

    assert result.status == "halted"
    assert result.termination["reason"] == "safety_contract_blocked_action"
    assert result.summary["termination"]["detail"]["action"] == "download_file"


@pytest.mark.asyncio
async def test_mission_executor_asks_on_uncertainty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = MissionSpec(
        instruction="Escalate on low confidence",
        execute=False,
        ask_on_uncertainty=True,
    )
    subgoal = MissionSubgoal(id="sg_low_conf", description="Check status", planner_metadata={})
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda _: [subgoal])

    async def fake_pipeline(self, **_: Dict[str, object]) -> Dict[str, object]:  # noqa: ARG001
        return {
            "completion": {"complete": True, "reason": "ok"},
            "artifacts": {},
            "run_context": {
                "page_intents": [
                    {
                        "intent": "listing_page",
                        "confidence": 0.2,
                        "strategy": "listing_extraction",
                        "signals": {},
                    }
                ]
            },
        }

    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", fake_pipeline)
    executor = MissionExecutor(
        settings={"planner": {}, "logging": {"artifact_root": str(tmp_path)}},
        artifacts_root=tmp_path,
    )
    result = await executor.run_mission(spec)

    assert result.status == "ask_human"
    assert result.summary["reason"] == "ask_on_uncertainty"
    assert result.summary["termination"]["reason"] == "low_confidence"
*** End File