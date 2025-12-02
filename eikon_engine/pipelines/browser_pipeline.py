"""High-level pipeline that wires planner, strategist, and browser worker."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from eikon_engine.config_loader import load_settings
from eikon_engine.core.orchestrator import build_orchestrator
from eikon_engine.core.strategist import Strategist
from eikon_engine.planning.memory_store import MemoryStore
from eikon_engine.planning.planner_online import OnlinePlanner
from eikon_engine.utils.logging_utils import ArtifactLogger
from eikon_engine.browser.worker_v1 import BrowserWorkerV1


async def run_browser_goal(goal: str, *, settings_path: Path | None = None) -> Dict[str, Any]:
    """Execute the browser pipeline end-to-end."""

    settings = load_settings(settings_path)
    memory = MemoryStore()
    planner = OnlinePlanner(memory_store=memory)
    strategist = Strategist(planner=planner)
    artifact_root = Path(settings.get("logging", {}).get("artifact_root", "artifacts"))
    logger = ArtifactLogger(root=artifact_root, prefix="browser")
    worker = BrowserWorkerV1(settings=settings, logger=logger)
    orchestrator = build_orchestrator(strategist=strategist, worker=worker, logger=logger, settings=settings)
    try:
        result = await orchestrator.run(goal)
        result.setdefault("artifacts", logger.to_dict())
        return result
    finally:
        await worker.close()
