from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from eikon_engine.core.completion import build_completion
from eikon_engine.core.goal_manager import GoalManager
from eikon_engine.core.orchestrator import Orchestrator
from eikon_engine.core.strategist import Strategist
from eikon_engine.planning.memory_store import MemoryStore
from eikon_engine.utils.logging_utils import ArtifactLogger


class StubPlanner:
    def __init__(self) -> None:
        self.memory_store = MemoryStore()

    async def create_plan(self, goal: str, *, last_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        _ = last_result
        return {
            "goal": goal,
            "actions": [
                {
                    "action": "noop",
                    "goal": goal,
                }
            ],
            "completion": build_completion(complete=True, reason="stub"),
        }


class StubWorker:
    def __init__(self, *, fail_on_attempt: Optional[int] = None) -> None:
        self.fail_on_attempt = fail_on_attempt
        self.logger = None
        self._calls = 0

    async def execute(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        self._calls += 1
        incomplete = self.fail_on_attempt is not None and self._calls >= self.fail_on_attempt
        completion = build_completion(complete=not incomplete, reason="done" if not incomplete else "pending")
        return {"completion": completion, "steps": [metadata]}

    async def close(self) -> None:  # pragma: no cover - stub helper
        return None


@pytest.mark.asyncio
async def test_run_multi_goal_completes_all(tmp_path: Path) -> None:
    planner = StubPlanner()
    strategist = Strategist(planner=planner, memory_store=planner.memory_store)
    worker = StubWorker()
    logger = ArtifactLogger(base_dir=tmp_path / "run")
    orchestrator = Orchestrator(strategist=strategist, worker=worker, logger=logger)

    manager = GoalManager.parse("Log in, extract title, logout")
    result = await orchestrator.run_multi_goal(goal_manager=manager)

    assert result["completion"]["complete"] is True
    assert len(result["goal_runs"]) == len(manager.goals)
    assert Path(result["goal_runs"][0]["artifacts"]["base_dir"]).exists()


@pytest.mark.asyncio
async def test_run_multi_goal_stops_on_incomplete(tmp_path: Path) -> None:
    planner = StubPlanner()
    strategist = Strategist(planner=planner, memory_store=planner.memory_store)
    worker = StubWorker(fail_on_attempt=2)
    logger = ArtifactLogger(base_dir=tmp_path / "run")
    orchestrator = Orchestrator(strategist=strategist, worker=worker, logger=logger)

    manager = GoalManager.parse("Log in, extract title, logout")
    result = await orchestrator.run_multi_goal(goal_manager=manager)

    assert result["completion"]["complete"] is False
    assert any(run["goal"] == "perform_login" for run in result["goal_runs"])
    assert len(result["goal_runs"]) < len(manager.goals)
