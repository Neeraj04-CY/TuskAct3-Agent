"""Browser pipeline that wires Planner v3, Strategist V2, and Orchestrator V2."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from eikon_engine.config_loader import load_settings
from eikon_engine.core.orchestrator_v2 import OrchestratorV2
from eikon_engine.planning.planner_v3 import plan_from_goal
from eikon_engine.strategist.strategist_v2 import StrategistV2
from eikon_engine.utils.logging_utils import ArtifactLogger
from eikon_engine.workers.browser_worker import BrowserWorker


logger = logging.getLogger(__name__)


class PlannerV3Adapter:
    """Async wrapper that exposes planner_v3 via Strategist's interface."""

    def __init__(self, *, context: Optional[Dict[str, Any]] = None) -> None:
        self._context = context or {}

    async def create_plan(self, goal: str, *, last_result: Dict[str, Any] | None = None) -> Dict[str, Any]:
        ctx = dict(self._context)
        if last_result:
            ctx["last_result"] = last_result
        return plan_from_goal(goal, context=ctx)


async def _run_once(
    goal_text: str,
    *,
    allow_sensitive: bool,
    dry_run: bool,
    settings: Dict[str, Any],
    artifact_prefix: str,
) -> Dict[str, Any]:
    planner_context = settings.get("planner", {})
    planner = PlannerV3Adapter(context=planner_context)
    strategist = StrategistV2(planner=planner)

    logging_cfg = settings.get("logging", {})
    artifact_root = Path(logging_cfg.get("artifact_root", "artifacts"))
    run_logger = ArtifactLogger(root=artifact_root, prefix=artifact_prefix or "browser_v2")

    worker = BrowserWorker(
        settings=settings,
        logger=run_logger,
        enable_playwright=False if dry_run else None,
    )
    orchestrator = OrchestratorV2(
        strategist=strategist,
        worker=worker,
        logger=run_logger,
    )

    logger.info("Running Strategist V2 pipeline", extra={"goal": goal_text, "dry_run": dry_run})
    try:
        result = await orchestrator.run_goal(goal_text)
    finally:
        await worker.close()

    metadata = result.setdefault("metadata", {})
    metadata["allow_sensitive"] = allow_sensitive
    metadata["dry_run"] = dry_run
    metadata["planner_context"] = planner_context
    result["artifacts"] = run_logger.to_dict()
    logger.info("Pipeline complete", extra={"completed": result.get("completion", {}).get("complete")})
    return result


def run_pipeline(
    goal_text: str,
    *,
    allow_sensitive: bool = False,
    dry_run: bool = True,
    settings: Optional[Dict[str, Any]] = None,
    artifact_prefix: str = "browser_v2",
) -> Dict[str, Any]:
    """Public entry point that executes the Strategist V2 pipeline."""

    resolved_settings = settings or load_settings()
    return asyncio.run(
        _run_once(
            goal_text,
            allow_sensitive=allow_sensitive,
            dry_run=dry_run,
            settings=resolved_settings,
            artifact_prefix=artifact_prefix,
        )
    )
