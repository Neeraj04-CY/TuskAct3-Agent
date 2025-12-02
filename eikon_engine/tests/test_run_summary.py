from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from eikon_engine.core.completion import build_completion
from eikon_engine.core.goal_manager import Goal, GoalManager
from eikon_engine.core.orchestrator import Orchestrator
from eikon_engine.core.strategist import Strategist
from eikon_engine.planning.memory_store import MemoryStore
from eikon_engine.utils.logging_utils import ArtifactLogger


class SummaryPlanner:
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


class SummaryWorker:
    def __init__(self) -> None:
        self.logger = None

    async def execute(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "steps": [metadata],
            "screenshots": [],
            "dom_snapshot": "<html></html>",
            "layout_graph": "root",
            "completion": build_completion(complete=True, reason="done", payload={"steps": 1}),
            "error": None,
        }

    async def close(self) -> None:  # pragma: no cover - stub
        return None


@pytest.mark.asyncio
async def test_run_summary_written(tmp_path: Path) -> None:
    planner = SummaryPlanner()
    strategist = Strategist(planner=planner, memory_store=planner.memory_store)
    worker = SummaryWorker()
    logger = ArtifactLogger(base_dir=tmp_path / "run")
    orchestrator = Orchestrator(strategist=strategist, worker=worker, logger=logger)

    goal = Goal(name="collect_data", description="Collect something")
    manager = GoalManager.from_goals(instruction="Collect data", goals=[goal])
    await orchestrator.run_multi_goal(goal_manager=manager)

    summary_path = logger.base_dir / "run_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["goals"]
    assert "final_completion_status" in summary
