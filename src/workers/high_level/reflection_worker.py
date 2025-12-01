"""Reflection worker that evaluates strategist outputs for autonomous mode."""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional

from src.strategist.strategist_v1 import GoalWorker


class ReflectionWorker(GoalWorker):
    """Determines whether a goal is complete and proposes follow-ups."""

    COMPLETION_KEYWORDS = ("summary", "done", "complete", "success")

    def __init__(self) -> None:
        self._previous_next_goal: Optional[str] = None
        self._repeat_count = 0

    async def run(self, description: str, prev_results: Dict[str, Any]) -> Dict[str, Any]:
        completion_meta = prev_results.get("completion")
        if isinstance(completion_meta, dict) and completion_meta.get("complete"):
            return {
                "goal_satisfied": True,
                "next_goal": None,
                "intent_corrected": False,
                "repair_suggestion": None,
                "notes": completion_meta.get("reason", "Completion signal received."),
                "stop_reason": completion_meta,
            }

        plan = prev_results.get("plan", {}) or {}
        nodes = plan.get("nodes") or []
        execution_results = prev_results.get("results", {}) or {}
        related_memories = prev_results.get("related_memories") or []

        notes: List[str] = []
        goal_satisfied = False
        intent_corrected = False
        repair_suggestion: Optional[str] = None

        goal_text = str(description).strip()
        if not goal_text:
            goal_text = "Clarify missing goal"
            intent_corrected = True
            notes.append("Goal missing; prompting clarification.")

        if len(goal_text.split()) < 5:
            intent_corrected = True
            goal_text = f"Clarify intent: {goal_text}".strip()
            notes.append("Goal was short; added clarification prompt.")

        if related_memories:
            hints = ", ".join(
                memory.get("metadata", {}).get("title", "memory")
                for memory in itertools.islice(related_memories, 3)
            )
            goal_text = f"Incorporate memory insights ({hints}). {goal_text}".strip()
            intent_corrected = True
            notes.append("Memory insights incorporated into goal.")

        for item in execution_results.values():
            if not item:
                continue
            text = str(item).lower()
            if any(keyword in text for keyword in self.COMPLETION_KEYWORDS):
                goal_satisfied = True
                notes.append("Detected completion keyword in results.")
                break
            if "error" in text or "exception" in text:
                repair_suggestion = f"Repair failure for goal: {description}"
                notes.append("Detected error; proposing repair goal.")

        if not execution_results:
            repair_suggestion = f"No results for goal: {description}. Regenerate plan."
            notes.append("No results found; proposing repair plan.")

        if len(nodes) <= 1 and execution_results:
            goal_satisfied = True
            notes.append("Plan had <= 1 node with results; assuming completion.")

        next_goal = None
        if goal_satisfied:
            notes.append("Marking goal satisfied.")
            self._previous_next_goal = None
            self._repeat_count = 0
        else:
            if repair_suggestion:
                next_goal = repair_suggestion
            else:
                next_goal = f"Refine: {goal_text}".strip()
            notes.append("Continuing with refined/repair goal.")

            if next_goal:
                if self._previous_next_goal == next_goal:
                    self._repeat_count += 1
                else:
                    self._repeat_count = 1
                    self._previous_next_goal = next_goal
            else:
                self._repeat_count = 0
                self._previous_next_goal = None

            if self._repeat_count >= 2:
                goal_satisfied = True
                next_goal = None
                notes.append("Next goal repeated twice; treating as satisfied.")
                self._previous_next_goal = None
                self._repeat_count = 0

        return {
            "goal_satisfied": goal_satisfied,
            "next_goal": next_goal,
            "intent_corrected": intent_corrected,
            "repair_suggestion": repair_suggestion,
            "notes": " ".join(notes),
        }
