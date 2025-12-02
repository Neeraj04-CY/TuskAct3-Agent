"""Public WebAgent API that powers autonomous browser runs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from eikon_engine.config_loader import load_settings
from eikon_engine.core.goal_manager import Goal, GoalManager
from eikon_engine.core.orchestrator import build_orchestrator
from eikon_engine.core.strategist import Strategist
from eikon_engine.planning.memory_store import MemoryStore
from eikon_engine.planning.planner_online import OnlinePlanner
from eikon_engine.utils.logging_utils import ArtifactLogger
from eikon_engine.workers.browser_worker import BrowserWorker


class WebAgent:
    """High-level facade that exposes a synchronous and async run API."""

    def __init__(self, *, settings_path: Path | None = None) -> None:
        self.settings = load_settings(settings_path)
        self.memory_store = MemoryStore()
        self.planner = OnlinePlanner(memory_store=self.memory_store)
        self.strategist = Strategist(planner=self.planner, memory_store=self.memory_store)
        self.worker = BrowserWorker(settings=self.settings)
        self.orchestrator = build_orchestrator(
            strategist=self.strategist,
            worker=self.worker,
            logger=None,
            settings=self.settings,
        )
        artifact_root = self.settings.get("logging", {}).get("artifact_root", "artifacts")
        self._artifact_root = Path(artifact_root)
        self._closed = False

    def run(self, instruction: str, *, multi_goal: bool = True) -> Dict[str, Any]:
        """Blocking helper that executes the agent and closes resources automatically."""

        async def _runner() -> Dict[str, Any]:
            try:
                return await self.run_async(instruction, multi_goal=multi_goal)
            finally:
                await self.close()

        return asyncio.run(_runner())

    async def run_async(self, instruction: str, *, multi_goal: bool = True) -> Dict[str, Any]:
        """Async entrypoint that keeps the agent open for multiple invocations."""

        manager = GoalManager.parse(instruction)
        if multi_goal:
            return await self.run_goal_manager(manager)
        logger = self._build_logger(goal_name=instruction)
        try:
            result = await self._run_with_logger(logger, single_goal=instruction)
        finally:
            self._detach_logger()
        result.setdefault("artifacts", logger.to_dict())
        return result

    async def run_goal_manager(self, goal_manager: GoalManager) -> Dict[str, Any]:
        """Execute a provided GoalManager."""

        logger = self._build_logger(goal_name="multi-goal")
        try:
            result = await self._run_with_logger(logger, goal_manager=goal_manager)
        finally:
            self._detach_logger()
        result.setdefault("artifacts", logger.to_dict())
        return result

    async def run_goals(
        self,
        goals: Sequence[Goal] | Iterable[Goal],
        *,
        instruction: str = "multi-goal run",
    ) -> Dict[str, Any]:
        """Helper that accepts a flat collection of Goal objects."""

        manager = GoalManager.from_goals(instruction=instruction, goals=list(goals))
        return await self.run_goal_manager(manager)

    async def close(self) -> None:
        if self._closed:
            return
        await self.worker.close()
        self._closed = True

    async def __aenter__(self) -> "WebAgent":  # pragma: no cover - convenience
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - convenience
        await self.close()

    def _build_logger(self, *, goal_name: str) -> ArtifactLogger:
        self._ensure_open()
        logger = ArtifactLogger(root=self._artifact_root, prefix="web_agent", goal_name=goal_name)
        self.worker.logger = logger
        self.orchestrator.logger = logger
        return logger

    async def _run_with_logger(
        self,
        logger: ArtifactLogger,
        *,
        goal_manager: GoalManager | None = None,
        single_goal: str | None = None,
    ) -> Dict[str, Any]:
        if goal_manager:
            return await self.orchestrator.run_multi_goal(goal_manager=goal_manager)
        if single_goal:
            return await self.orchestrator.run_single_goal(single_goal)
        raise ValueError("Either goal_manager or single_goal must be provided")

    def _detach_logger(self) -> None:
        self.worker.logger = None
        self.orchestrator.logger = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("WebAgent has been closed. Create a new instance for another run.")