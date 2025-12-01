from __future__ import annotations

from typing import Iterable, List

from src.common_types import WorkflowStep


class SimpleScheduler:
    """
    Placeholder scheduler.

    Future:
    - topological sort based on depends_on
    - parallel groups
    - dynamic priorities
    """

    def order_steps(self, steps: Iterable[WorkflowStep]) -> List[WorkflowStep]:
        return list(steps)