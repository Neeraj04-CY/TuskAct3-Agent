from __future__ import annotations

from typing import Any, Dict

from src.strategist.strategist_v1 import GoalWorker


class AnalyzerWorker(GoalWorker):
    """
    Very simple analyzer: echoes the description and previous context size.

    Later this can:
    - Use LLM to analyze the goal.
    - Suggest sub-goals and constraints.
    """

    async def run(self, step_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "analysis": step_description,
            "previous_steps": list(context.keys()),
        }