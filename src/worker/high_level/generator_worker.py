from __future__ import annotations

from typing import Any, Dict

from src.strategist.strategist_v1 import GoalWorker


class GeneratorWorker(GoalWorker):
    """
    Very simple generator: builds a structured response from preceding analysis.

    In the future this will be LLM-based summarization / reasoning.
    """

    async def run(self, step_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        analysis = context.get("step_1", {})
        return {
            "description": step_description,
            "summary": f"Generated response based on analysis: {analysis!r}",
        }