from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List

from src.common_types import RiskItem, WorkflowObject, WorkflowStep
from src.strategist.parser import RequestParser, SimpleRequestParser
from src.strategist.planner import TaskPlanner, HeuristicTaskPlanner
from src.strategist.risk_analyzer import RiskAnalyzer, SimpleRiskAnalyzer
from src.strategist.tool_selector import ToolSelector, SimpleToolSelector


@dataclass
class StrategistConfig:
    """
    Config structure for the Strategist.
    Extend this as configuration grows (e.g., model names, depth limits).
    """
    max_steps: int = 32


class Strategist:
    """
    High-level Strategist module.

    Responsibilities:
    - Parse user request.
    - Analyze complexity (placeholder for now).
    - Plan tasks into steps.
    - Select tools and skills.
    - Query memory (hook for Memory Engine).
    - Perform risk analysis.
    - Generate final WorkflowObject.

    This class is wired with dependencies via constructor (dependency injection).
    """

    def __init__(
        self,
        parser: RequestParser | None = None,
        planner: TaskPlanner | None = None,
        tool_selector: ToolSelector | None = None,
        risk_analyzer: RiskAnalyzer | None = None,
        config: StrategistConfig | None = None
    ) -> None:
        self._parser = parser or SimpleRequestParser()
        self._planner = planner or HeuristicTaskPlanner()
        self._tool_selector = tool_selector or SimpleToolSelector()
        self._risk_analyzer = risk_analyzer or SimpleRiskAnalyzer()
        self._config = config or StrategistConfig()

    def create_workflow(self, user_input: str) -> WorkflowObject:
        """
        Main entry point: builds a WorkflowObject from raw user input.
        """
        parsed = self._parser.parse(user_input)
        task = parsed["task"]

        execution_plan: List[WorkflowStep] = self._planner.plan(task)
        tools_needed = self._tool_selector.select_tools(task)
        skills_to_load = self._tool_selector.select_skills(task)
        risk_analysis: List[RiskItem] = self._risk_analyzer.analyze(task)

        workflow_id = str(uuid.uuid4())

        return WorkflowObject(
            workflow_id=workflow_id,
            task=task,
            subtasks=[step.description for step in execution_plan],
            tools_needed=tools_needed,
            skills_to_load=skills_to_load,
            memory_references=[],  # v1: memory not wired yet
            constraints=[],        # later: from config or user
            risk_analysis=risk_analysis,
            success_criteria=["Task description is fulfilled and output is coherent."],
            estimated_time="unknown",
            execution_plan=execution_plan
        )