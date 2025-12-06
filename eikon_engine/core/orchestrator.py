"""Task orchestrator responsible for running strategy steps."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING

from eikon_engine.core.completion import build_completion, is_complete
from eikon_engine.core.goal_manager import Goal, GoalManager
from eikon_engine.core.types import CompletionPayload
from eikon_engine.utils.logging_utils import ArtifactLogger

if TYPE_CHECKING:  # pragma: no cover
    from eikon_engine.core.strategist import Strategist
    from eikon_engine.workers.browser_worker import BrowserWorker


@dataclass
class Orchestrator:
    """Executes strategy steps until completion criteria are met."""

    strategist: "Strategist"
    worker: "BrowserWorker"
    max_steps: int = 25
    retry_limit: int = 3
    logger: ArtifactLogger | None = None
    transcript: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._reflections: int = 0

    async def run(self, goal: str) -> Dict[str, Any]:
        """Drive a single goal for backwards compatibility."""

        return await self.run_single_goal(goal)

    async def run_single_goal(
        self,
        goal: str,
        *,
        logger: ArtifactLogger | None = None,
        write_summary: bool = True,
    ) -> Dict[str, Any]:
        """Execute a single goal with optional goal-specific logging."""

        self.transcript = []
        steps_executed = 0
        retries = 0
        goal_logger = self._resolve_goal_logger(goal, override=logger)
        start_time = datetime.now(UTC)
        worker_logger_applied = False
        previous_worker_logger: ArtifactLogger | None = None
        if goal_logger and hasattr(self.worker, "logger"):
            previous_worker_logger = getattr(self.worker, "logger")
            setattr(self.worker, "logger", goal_logger)
            worker_logger_applied = True

        await self.strategist.initialize(goal)
        await self.strategist.ensure_plan()

        try:
            while steps_executed < self.max_steps:
                if not self.strategist.has_next():
                    await self.strategist.ensure_plan()
                    if not self.strategist.has_next():
                        break
                step = self.strategist.next_step()
                action_meta = step.metadata.get("action")
                action_repr = str(action_meta)
                action_url = action_meta.get("url") if isinstance(action_meta, dict) else None
                step_idx = goal_logger.log_step(step.metadata, goal=goal) if goal_logger else None
                worker_metadata = dict(step.metadata)
                worker_metadata.setdefault("goal", goal)
                try:
                    result = await self.worker.execute(worker_metadata)
                except Exception as exc:  # noqa: BLE001
                    retries += 1
                    if retries > self.retry_limit:
                        completion = build_completion(
                            complete=True,
                            reason="retry limit exceeded",
                            payload={"error": str(exc)},
                        )
                        failure = {"error": str(exc), "completion": completion}
                        self.transcript.append({"step": step.metadata, "result": failure})
                        if goal_logger and step_idx is not None:
                            goal_logger.log_trace(
                                goal=goal,
                                step_index=step_idx,
                                action=action_repr,
                                url=action_url,
                                completion=completion,
                            )
                        break
                    continue

                retries = 0
                steps_executed += 1
                await self.strategist.record_result(result)
                await self.strategist.ensure_plan()
                self.transcript.append({"step": step.metadata, "result": result})
                if goal_logger and step_idx is not None:
                    goal_logger.log_trace(
                        goal=goal,
                        step_index=step_idx,
                        action=action_repr,
                        url=action_url,
                        completion=result.get("completion"),
                    )
                if is_complete(result):
                    break
        finally:
            if worker_logger_applied:
                setattr(self.worker, "logger", previous_worker_logger)

        completion = self.strategist.completion_state()
        if not completion["complete"]:
            completion = build_completion(complete=False, reason="max steps reached", payload={})
        response = {
            "goal": goal,
            "steps": self.transcript,
            "completion": completion,
        }
        if goal_logger:
            response["artifacts"] = goal_logger.to_dict()
        if write_summary:
            end_time = datetime.now(UTC)
            error_reason = None
            if not completion["complete"]:
                if self.transcript:
                    error_reason = self.transcript[-1]["result"].get("error")
                error_reason = error_reason or completion.get("reason")
            summary_goals = [
                {
                    "goal_id": 1,
                    "goal_str": goal,
                    "status": "completed" if completion["complete"] else "incomplete",
                    "error_reason": error_reason,
                }
            ]
            self._record_run_summary(
                start_time=start_time,
                end_time=end_time,
                goals=summary_goals,
                total_steps=steps_executed,
                completion_status=completion["complete"],
            )
        return response

    async def run_multi_goal(
        self,
        goals: Sequence[Goal] | Sequence[Dict[str, Any]] | None = None,
        *,
        instruction: str | None = None,
        goal_manager: GoalManager | None = None,
    ) -> Dict[str, Any]:
        """Execute multiple derived goals sequentially until completion."""

        manager = self._resolve_goal_manager(instruction=instruction, manager=goal_manager, goals=goals)
        goal_runs: List[Dict[str, Any]] = []
        index = 1

        start_time = datetime.now(UTC)
        while True:
            next_goal = manager.next_goal()
            if not next_goal:
                break
            goal_logger = self._goal_child_logger(next_goal, index=index)
            instruction_text = next_goal.description or next_goal.name
            result = await self.run_single_goal(instruction_text, logger=goal_logger, write_summary=False)
            result["goal"] = next_goal.name
            result["metadata"] = next_goal.metadata
            goal_runs.append(result)
            manager.update(result)
            index += 1
            if not result.get("completion", {}).get("complete"):
                break

        report = manager.progress_report()
        report["goal_runs"] = goal_runs
        if self.logger:
            report["artifacts"] = self.logger.to_dict()
        end_time = datetime.now(UTC)
        goals_summary = self._summarize_manager_goals(manager)
        total_steps = sum(len(run.get("steps", [])) for run in goal_runs)
        self._record_run_summary(
            start_time=start_time,
            end_time=end_time,
            goals=goals_summary,
            total_steps=total_steps,
            completion_status=report["completion"]["complete"],
        )
        return report

    def _resolve_goal_logger(self, goal: str, *, override: ArtifactLogger | None = None) -> ArtifactLogger | None:
        if override:
            return override
        if not self.logger:
            return None
        return self.logger.create_child(_safe_goal_dir(goal), goal_name=goal)

    def _goal_child_logger(self, goal: Goal, *, index: int) -> ArtifactLogger | None:
        if not self.logger:
            return None
        return self.logger.create_child(_default_child_name(goal, index=index), goal_name=goal.name)

    def _resolve_goal_manager(
        self,
        *,
        instruction: Optional[str],
        manager: GoalManager | None,
        goals: Sequence[Goal] | Sequence[Dict[str, Any]] | None,
    ) -> GoalManager:
        if manager:
            return manager
        if goals:
            goal_objs: List[Goal] = []
            for entry in goals:
                if isinstance(entry, Goal):
                    goal_objs.append(entry)
                else:
                    goal_objs.append(Goal(name=str(entry.get("goal")), description=entry.get("description")))
            return GoalManager.from_goals(instruction=instruction or "multi-goal run", goals=goal_objs)
        if instruction:
            return GoalManager.parse(instruction)
        raise ValueError("instruction or goal_manager must be provided for multi-goal runs")

    def _record_run_summary(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        goals: List[Dict[str, Any]],
        total_steps: int,
        completion_status: bool,
    ) -> None:
        logger = self.logger
        if not logger:
            return
        payload = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "goals": goals,
            "total_steps_executed": total_steps,
            "total_reflections": self._reflections,
            "final_completion_status": completion_status,
            "run_dir": str(logger.base_dir),
        }
        logger.write_summary(payload)

    def _summarize_manager_goals(self, manager: GoalManager) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for idx, goal in enumerate(manager.goals, start=1):
            status = "completed" if goal.status == "complete" else "incomplete"
            error_reason = None
            if status == "incomplete" and goal.result:
                completion = goal.result.get("completion", {})
                error_reason = goal.result.get("error") or completion.get("reason")
            entries.append(
                {
                    "goal_id": idx,
                    "goal_str": goal.description or goal.name,
                    "status": status,
                    "error_reason": error_reason,
                }
            )
        return entries


def build_orchestrator(
    *,
    strategist: "Strategist",
    worker: "BrowserWorker",
    logger: ArtifactLogger | None = None,
    settings: Dict[str, Any] | None = None,
) -> Orchestrator:
    """Factory for Orchestrator that reads limits from settings."""

    completion_config = (settings or {}).get("completion", {})
    return Orchestrator(
        strategist=strategist,
        worker=worker,
        max_steps=int(completion_config.get("max_steps", 25)),
        retry_limit=int(completion_config.get("retry_limit", 3)),
        logger=logger,
    )


def _safe_goal_dir(goal: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", goal.lower()).strip("_")
    return slug or "goal"


def _default_child_name(goal: Goal, *, index: int) -> str:
    slug = goal.metadata.get("slug") or _safe_goal_dir(goal.name)
    return f"goal_{index:02d}_{slug}"
