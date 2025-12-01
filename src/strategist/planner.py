from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.common_types import WorkflowStep


class TaskPlanner(ABC):
    """
    Produces a hierarchical execution plan (list of WorkflowSteps) from a task description.

    Responsibilities:
    - Task decomposition into subtasks and steps.
    - Basic dependency graph (depends_on).
    """

    @abstractmethod
    def plan(self, task_description: str) -> List[WorkflowStep]:
        raise NotImplementedError


class HeuristicTaskPlanner(TaskPlanner):
    """
    Very simple v1 planner that creates a single-step plan.
    This is a placeholder for more advanced LLM or heuristic planning.
    """

    def plan(self, task_description: str) -> List[WorkflowStep]:
        return [
            WorkflowStep(
                step_id="step-1",
                description=task_description,
                tool=None,
                skill=None,
                inputs={"task": task_description},
                depends_on=[]
            )
        ]