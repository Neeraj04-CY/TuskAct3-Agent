from __future__ import annotations

import pytest

from eikon_engine.core.completion import build_completion
from eikon_engine.core.goal_manager import GoalManager
from eikon_engine.core.orchestrator import Orchestrator
from eikon_engine.core.strategist import Strategist
from eikon_engine.planning.planner_offline import OfflinePlanner
from eikon_engine.utils.logging_utils import ArtifactLogger


class _StubWorker:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(self, metadata: dict) -> dict:
        self.calls.append(metadata)
        return {"completion": build_completion(complete=True, reason="stub"), "metadata": metadata}

    async def close(self) -> None:  # pragma: no cover - compatibility hook
        return None


@pytest.mark.asyncio
async def test_run_single_goal_creates_child_artifacts(tmp_path):
    planner = OfflinePlanner()
    strategist = Strategist(planner=planner)
    worker = _StubWorker()
    logger = ArtifactLogger(root=tmp_path, prefix="test_single")
    orchestrator = Orchestrator(strategist=strategist, worker=worker, logger=logger, max_steps=3)

    result = await orchestrator.run_single_goal("open the heroku login page")

    assert result["completion"]["complete"] is True
    assert worker.calls, "worker should have executed at least one step"
    assert "artifacts" in result
    assert result["artifacts"]["base_dir"].startswith(str(tmp_path))


@pytest.mark.asyncio
async def test_run_multi_goal_tracks_progress(tmp_path):
    planner = OfflinePlanner()
    strategist = Strategist(planner=planner)
    worker = _StubWorker()
    logger = ArtifactLogger(root=tmp_path, prefix="test_multi")
    orchestrator = Orchestrator(strategist=strategist, worker=worker, logger=logger, max_steps=3)

    manager = GoalManager.parse("Login to the demo app and capture the secure message, then logout.")
    report = await orchestrator.run_multi_goal(goal_manager=manager)

    assert report["completion"]["complete"] is True
    assert len(report["goal_runs"]) >= 2
    goal_dirs = logger.base_dir.glob("*/")
    assert any(child.is_dir() for child in goal_dirs)