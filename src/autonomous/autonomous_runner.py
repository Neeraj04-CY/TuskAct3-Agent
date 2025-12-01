"""Autonomous execution loop that chains strategist goals."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.workers.high_level.reflection_worker import ReflectionWorker
from src.runtime_state import RuntimeState, RuntimeStateMachine
from src.telemetry import Telemetry


class AutonomousRunner:
    """Self-healing autonomous loop that iteratively refines goals."""

    def __init__(
        self,
        strategist,
        max_iters: int = 5,
        reflection_worker: Optional[ReflectionWorker] = None,
        telemetry: Optional[Telemetry] = None,
    ) -> None:
        if max_iters < 1:
            raise ValueError("max_iters must be >= 1")
        self.strategist = strategist
        self.max_iters = max_iters
        self.reflection_worker = reflection_worker or ReflectionWorker()
        memory_manager = getattr(strategist, "memory_manager", None)
        self.telemetry = telemetry or Telemetry(memory_manager)

    async def run(self, initial_goal: str) -> Dict[str, Any]:
        """Execute the strategist repeatedly until satisfied or max_iters reached."""

        goal = initial_goal.strip()
        if not goal:
            raise ValueError("initial_goal cannot be empty")

        goals_run: List[str] = []
        trace: List[Dict[str, Any]] = []
        repeated_reflections = 0
        last_next_goal: Optional[str] = None
        state_machine = RuntimeStateMachine()

        for _ in range(self.max_iters):
            iteration = len(goals_run)
            state_machine.next(RuntimeState.PLANNING, {"iteration": iteration, "goal": goal})
            state_machine.next(RuntimeState.EXECUTING, {"iteration": iteration})
            goals_run.append(goal)
            try:
                result = await self.strategist.run(goal)
            except Exception as exc:  # pragma: no cover - defensive guard
                trace.append({
                    "goal": goal,
                    "error": str(exc),
                })
                break

            state_machine.next(RuntimeState.REFLECTING, {"iteration": iteration})
            reflection = await self.reflection_worker.run(goal, result)
            await self.telemetry.trace_event("reflection_created", {"goal": goal, "notes": reflection.get("notes")})

            if last_next_goal and reflection.get("next_goal") == last_next_goal:
                repeated_reflections += 1
            else:
                repeated_reflections = 0
                last_next_goal = reflection.get("next_goal")

            results_map = result.get("results") or {}
            has_errors = any("error" in str(output).lower() for output in results_map.values())
            results_empty = not results_map
            self_healing = bool(reflection.get("repair_suggestion")) or has_errors or results_empty

            next_goal = reflection.get("next_goal")
            loop_detected = bool(next_goal and next_goal in goals_run)

            trace_entry = {
                "goal": goal,
                "parsed_goal": result.get("parsed_goal"),
                "plan": result.get("plan"),
                "assignments": result.get("assignments"),
                "results": results_map,
                "reflection": reflection,
                "next_goal": next_goal,
                "self_healing": self_healing,
                "intent_corrected": reflection.get("intent_corrected", False),
                "completion": result.get("completion"),
            }

            if repeated_reflections >= 1:
                loop_detected = True

            trace_entry["loop_detected"] = loop_detected
            trace.append(trace_entry)

            if reflection.get("goal_satisfied"):
                break

            if not next_goal:
                break

            if loop_detected:
                await self.telemetry.trace_event("loop_detected", {"goal": next_goal or goal})
                break

            goal = next_goal
            state_machine.next(RuntimeState.RETRYING, {"iteration": iteration, "next_goal": goal})

        final_goal = goals_run[-1] if goals_run else initial_goal
        state_machine.next(RuntimeState.HALT, {"iterations": len(goals_run)})
        return {
            "initial_goal": initial_goal,
            "final_goal": final_goal,
            "iterations": len(goals_run),
            "goals_run": goals_run,
            "trace": trace,
        }
