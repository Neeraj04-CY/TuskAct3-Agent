"""Mission execution orchestration built on Strategist V2."""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from eikon_engine.config_loader import load_settings
from eikon_engine.core.orchestrator_v2 import OrchestratorV2
from eikon_engine.missions.mission_planner import MissionPlanningError, plan_mission
from eikon_engine.missions.mission_schema import (
    MissionResult,
    MissionSpec,
    MissionStatus,
    MissionSubgoal,
    MissionSubgoalResult,
)
from eikon_engine.pipelines.browser_pipeline import PlannerV3Adapter
from eikon_engine.strategist.agent_memory import AgentMemory
from eikon_engine.strategist.strategist_v2 import StrategistV2
from eikon_engine.workers.browser_worker import BrowserWorker
from .artifacts import MissionArtifactLogger

SleepFn = Callable[[float], Awaitable[None]]


class MissionExecutor:
    """High-level mission orchestrator built on the browser pipeline."""

    def __init__(
        self,
        *,
        settings: Optional[Dict[str, Any]] = None,
        artifacts_root: Path | str | None = None,
        memory_manager: Optional[AgentMemory] = None,
        logger: Optional[logging.Logger] = None,
        sleep_fn: Optional[SleepFn] = None,
        debug_browser: bool = False,
    ) -> None:
        self.settings = settings or load_settings()
        self.artifacts_root = Path(artifacts_root or Path("artifacts"))
        self.memory_manager = memory_manager
        self.logger = logger or logging.getLogger(__name__)
        self._sleep = sleep_fn or asyncio.sleep
        self.debug_browser = debug_browser

    async def run_mission(self, mission_spec: MissionSpec) -> MissionResult:
        """Execute all mission subgoals sequentially."""

        start_ts = datetime.now(UTC)
        mission_dir = self._build_mission_dir(mission_spec, start_ts)
        subgoal_results: List[MissionSubgoalResult] = []
        summary: Dict[str, Any] = {}
        status: MissionStatus = "running"
        worker: BrowserWorker | None = None
        try:
            try:
                subgoals = plan_mission(mission_spec)
            except MissionPlanningError as exc:
                status = "failed"
                summary = {"reason": "planner_error", "detail": str(exc)}
                end_ts = datetime.now(UTC)
                result = MissionResult(
                    mission_id=mission_spec.id,
                    status=status,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    subgoal_results=subgoal_results,
                    summary=summary,
                    artifacts_path=str(mission_dir),
                )
                self._write_result_file(mission_dir, result)
                return result

            worker = self._build_worker(mission_spec)
            deadline = start_ts + timedelta(seconds=mission_spec.timeout_secs)
            for index, subgoal in enumerate(subgoals, start=1):
                if datetime.now(UTC) > deadline:
                    status = "failed"
                    summary = {"reason": "timeout", "failed_subgoal": subgoal.id}
                    break
                subgoal_dir = mission_dir / f"subgoal_{index:02d}"
                subgoal_dir.mkdir(parents=True, exist_ok=True)
                result = await self._execute_subgoal(mission_spec, subgoal, subgoal_dir, worker)
                subgoal_results.append(result)
                secure_artifact = result.artifacts.get("secure_area") if result.artifacts else None
                if secure_artifact and secure_artifact.get("detected"):
                    status = "complete"
                    summary = {
                        "reason": "secure_area_detected",
                        "subgoals_completed": len(subgoal_results),
                        "secure_area": secure_artifact,
                    }
                    break  # secure area success termination condition
                if result.status != "complete":
                    status = "failed"
                    summary = {
                        "reason": "subgoal_failed",
                        "failed_subgoal": subgoal.id,
                        "error": result.error,
                    }
                    break
            else:
                if status == "running":
                    status = "complete"
                    summary = {
                        "reason": "mission_complete",
                        "subgoals_completed": len(subgoal_results),
                    }
            end_ts = datetime.now(UTC)
            mission_result = MissionResult(
                mission_id=mission_spec.id,
                status=status,
                start_ts=start_ts,
                end_ts=end_ts,
                subgoal_results=subgoal_results,
                summary=summary,
                artifacts_path=str(mission_dir),
            )
            self._write_result_file(mission_dir, mission_result)
            self._record_memory(summary)
            return mission_result
        finally:
            if worker:
                if self.debug_browser and mission_spec.execute:
                    message = "[DEBUG] Browser staying open for manual inspection"
                    self.logger.info(message)
                    print(message)
                    await getattr(worker, "await_manual_close", worker.shutdown)()
                else:
                    await worker.shutdown()

    def _build_mission_dir(self, mission_spec: MissionSpec, start_ts: datetime) -> Path:
        slug = _slugify(mission_spec.instruction)[:40]
        mission_dir = self.artifacts_root / f"mission_{start_ts.strftime('%Y%m%d_%H%M%S')}_{slug}"
        mission_dir.mkdir(parents=True, exist_ok=True)
        return mission_dir

    async def _execute_subgoal(
        self,
        mission_spec: MissionSpec,
        subgoal: MissionSubgoal,
        subgoal_dir: Path,
        worker: BrowserWorker,
    ) -> MissionSubgoalResult:
        attempts = 0
        last_error: str | None = None
        completion_payload: Dict[str, Any] | None = None
        artifacts: Dict[str, Any] = {}
        start_time = datetime.now(UTC)
        bootstrap_actions = subgoal.planner_metadata.get("bootstrap_actions") if worker else None
        while attempts <= mission_spec.max_retries:
            attempts += 1
            try:
                if bootstrap_actions:
                    payload = await self._execute_bootstrap_actions(
                        goal_text=subgoal.description,
                        actions=bootstrap_actions,
                        subgoal_dir=subgoal_dir,
                        worker=worker,
                    )
                else:
                    payload = await self._run_subgoal_pipeline(
                        goal_text=subgoal.description,
                        mission_instruction=mission_spec.instruction,
                        dry_run=not mission_spec.execute,
                        subgoal_dir=subgoal_dir,
                        allow_sensitive=mission_spec.allow_sensitive,
                        worker=worker,
                    )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                completion_payload = None
                artifacts = {"traceback": traceback.format_exc()}
            else:
                completion_payload = payload.get("completion")
                artifacts = dict(payload.get("artifacts", {}))
                error = payload.get("error")
                secure_payload = payload.get("secure_area")
                if secure_payload:
                    artifacts["secure_area"] = secure_payload
                last_error = error or completion_payload.get("reason") if completion_payload else error
                if completion_payload and completion_payload.get("complete") and not error:
                    last_error = None
                    break
            if attempts > mission_spec.max_retries:
                break
            if mission_spec.execute:
                delay = 2 ** (attempts - 1)
                await self._sleep(delay)
        end_time = datetime.now(UTC)
        status: MissionStatus = "complete" if last_error is None else "failed"
        return MissionSubgoalResult(
            subgoal_id=subgoal.id,
            description=subgoal.description,
            status=status,
            attempts=attempts,
            started_at=start_time,
            ended_at=end_time,
            completion=completion_payload,
            error=last_error,
            artifacts=artifacts,
        )

    async def _run_subgoal_pipeline(
        self,
        *,
        goal_text: str,
        mission_instruction: str,
        dry_run: bool,
        subgoal_dir: Path,
        allow_sensitive: bool,
        worker: BrowserWorker,
    ) -> Dict[str, Any]:
        planner_context = self.settings.get("planner", {})
        planner = PlannerV3Adapter(context=planner_context)
        strategist = StrategistV2(planner=planner)
        run_logger = MissionArtifactLogger(base_dir=subgoal_dir, goal_name=goal_text)
        worker.logger = run_logger
        worker.set_mission_context(
            mission_instruction=mission_instruction,
            subgoal_description=goal_text,
        )
        orchestrator = OrchestratorV2(
            strategist=strategist,
            worker=worker,
            logger=run_logger,
        )
        result = await orchestrator.run_goal(goal_text)
        metadata = result.setdefault("metadata", {})
        metadata["allow_sensitive"] = allow_sensitive
        metadata["dry_run"] = dry_run
        result["artifacts"] = run_logger.to_dict()
        return result

    async def _execute_bootstrap_actions(
        self,
        *,
        goal_text: str,
        actions: List[Dict[str, Any]],
        subgoal_dir: Path,
        worker: BrowserWorker,
    ) -> Dict[str, Any]:
        run_logger = MissionArtifactLogger(base_dir=subgoal_dir, goal_name=goal_text)
        worker.logger = run_logger
        result = await worker.execute({"action": actions, "goal": goal_text})
        payload = {
            "completion": result.get("completion"),
            "error": result.get("error"),
            "artifacts": run_logger.to_dict(),
        }
        return payload

    def _build_worker(self, mission_spec: MissionSpec) -> BrowserWorker:
        dry_run = not mission_spec.execute
        return BrowserWorker(
            settings=self.settings,
            logger=None,
            enable_playwright=False if dry_run else None,
            show_browser=None if dry_run else True,
        )

    def _write_result_file(self, mission_dir: Path, result: MissionResult) -> None:
        payload = result.model_dump(mode="json")
        (mission_dir / "mission_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _record_memory(self, summary: Dict[str, Any]) -> None:
        if not self.memory_manager:
            return
        add_memory = getattr(self.memory_manager, "add_memory", None)
        if callable(add_memory):
            try:
                add_memory(summary)
            except Exception:  # pragma: no cover - defensive logging
                self.logger.warning("mission memory write failed", exc_info=True)


def run_mission_sync(mission_spec: MissionSpec, executor: MissionExecutor | None = None) -> MissionResult:
    """Convenience wrapper for synchronous callers (e.g., CLI)."""

    executor = executor or MissionExecutor()
    return asyncio.run(executor.run_mission(mission_spec))


def _slugify(value: str) -> str:
    cleaned = [ch if ch.isalnum() else "_" for ch in value.lower()]
    slug = "".join(cleaned).strip("_")
    return slug or "mission"


__all__ = ["MissionExecutor", "run_mission_sync"]
