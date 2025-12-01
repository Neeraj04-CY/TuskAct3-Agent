from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from src.completion import extract_completion_metadata
from src.runtime_state import RuntimeState, RuntimeStateMachine
from src.telemetry import Telemetry
from src.tool_policy import ToolPolicy
from src.workers.high_level.reflection_worker import ReflectionWorker

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.memory.memory_manager import MemoryManager


class TaskOrchestrator:
    """High-level orchestrator for adaptive task execution."""

    def __init__(
        self,
        strategist,
        worker_registry: Optional[Dict[str, Any]] = None,
        memory_manager: Optional["MemoryManager"] = None,
        reflection_worker: Optional[ReflectionWorker] = None,
        tool_policy: Optional[ToolPolicy] = None,
        telemetry: Optional[Telemetry] = None,
    ) -> None:
        self.strategist = strategist
        self.worker_registry = worker_registry or getattr(strategist, "worker_registry", {})
        self.memory_manager = memory_manager or getattr(strategist, "memory_manager", None)
        self.reflection_worker = reflection_worker or ReflectionWorker()
        allowlist: Set[str] = set(self.worker_registry or {})
        self.tool_policy = tool_policy or ToolPolicy(allowed_workers=allowlist)
        self.telemetry = telemetry or Telemetry(self.memory_manager)

    async def execute(self, goal: str, max_iters: int = 10) -> Dict[str, Any]:
        transcript: List[Dict[str, Any]] = []
        seen_goals: Set[str] = set()
        current_goal = goal.strip()
        if not current_goal:
            raise ValueError("Goal cannot be empty")
        seen_goals.add(current_goal)

        state_machine = RuntimeStateMachine()
        status = "success"
        summary_notes: List[str] = []

        completion_meta: Optional[Dict[str, Any]] = None
        for iteration in range(max_iters):
            state_machine.next(RuntimeState.PLANNING, {"iteration": iteration, "goal": current_goal})
            try:
                plan_bundle = await self.strategist.plan(current_goal)
            except Exception as exc:  # pragma: no cover - defensive
                status = "error"
                await self.telemetry.trace_event(
                    "error_detected",
                    {"goal": current_goal, "reason": str(exc)},
                )
                transcript.append(
                    {
                        "goal": current_goal,
                        "plan": None,
                        "result": {"error": str(exc)},
                        "reflection": {},
                        "memory_written": None,
                    }
                )
                break
            plan_dict = plan_bundle["plan"]
            dag = plan_bundle["dag"]
            await self.telemetry.trace_event("plan_created", {"goal": current_goal, "nodes": len(plan_dict.get("nodes", []))})

            state_machine.next(RuntimeState.EXECUTING, {"iteration": iteration})
            try:
                level_results = await self.strategist.execute_dag(
                    dag,
                    self.worker_registry,
                    tool_policy=self.tool_policy,
                    telemetry=self.telemetry,
                    iteration=iteration,
                )
            except Exception as exc:  # pragma: no cover - defensive
                status = "error"
                await self.telemetry.trace_event(
                    "error_detected",
                    {"goal": current_goal, "reason": str(exc)},
                )
                transcript.append(
                    {
                        "goal": current_goal,
                        "plan": plan_dict,
                        "result": {"error": str(exc)},
                        "reflection": {},
                        "memory_written": None,
                    }
                )
                break

            payload = {
                "plan": plan_dict,
                "results": level_results,
                "related_memories": plan_bundle.get("parsed_goal", {}).get("related_memories"),
            }

            completion_candidate = extract_completion_metadata(level_results)
            if completion_candidate:
                payload["completion"] = completion_candidate
                completion_meta = completion_candidate

            state_machine.next(RuntimeState.REFLECTING, {"iteration": iteration})
            reflection = await self.reflection_worker.run(current_goal, payload)
            await self.telemetry.trace_event("reflection_created", {"goal": current_goal, "notes": reflection.get("notes")})

            memory_written = None
            if self.memory_manager:
                summary_block = f"Goal: {current_goal}\nReflection: {reflection}"
                try:
                    memory_written = await self.memory_manager.add_memory(
                        event_type="orchestrator_trace",
                        title=f"Task transcript {iteration+1}",
                        text=summary_block,
                        context={"goal": current_goal, "iteration": iteration},
                    )
                    await self.telemetry.trace_event("memory_written", {"memory_id": memory_written})
                except Exception as exc:  # pragma: no cover - defensive
                    await self.telemetry.trace_event(
                        "error_detected",
                        {"goal": current_goal, "reason": str(exc)},
                    )

            transcript.append(
                {
                    "goal": current_goal,
                    "plan": plan_dict,
                    "result": level_results,
                    "reflection": reflection,
                    "memory_written": memory_written,
                    "completion": completion_candidate,
                }
            )
            summary_notes.append(reflection.get("notes", ""))

            if completion_candidate:
                status = "completed"
                break

            if reflection.get("goal_satisfied"):
                break

            next_goal = reflection.get("next_goal")
            if not next_goal:
                break

            if next_goal in seen_goals:
                status = "loop_detected"
                await self.telemetry.trace_event("loop_detected", {"goal": next_goal})
                break

            seen_goals.add(next_goal)
            state_machine.next(RuntimeState.RETRYING, {"iteration": iteration, "next_goal": next_goal})
            current_goal = next_goal
        else:
            status = "max_iters"

        halt_context: Dict[str, Any] = {"iterations": len(transcript)}
        if completion_meta:
            halt_context["completion"] = completion_meta
        state_machine.next(RuntimeState.HALT, halt_context)
        summary = "\n".join(note for note in summary_notes if note)
        return {
            "status": status,
            "transcript": transcript,
            "summary": summary,
            "completion": completion_meta,
        }
