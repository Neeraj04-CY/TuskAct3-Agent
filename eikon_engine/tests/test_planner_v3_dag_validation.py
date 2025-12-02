from __future__ import annotations

import pytest

from eikon_engine.planning.planner_v3 import Task, TaskInput, validate_dag


def test_cycle_detection() -> None:
    tasks: list[Task] = [
        Task(id="task_1", tool="BrowserWorker", inputs=TaskInput(actions=[]), depends_on=["task_2"]),
        Task(id="task_2", tool="BrowserWorker", inputs=TaskInput(actions=[]), depends_on=["task_1"]),
    ]
    with pytest.raises(ValueError):
        validate_dag(tasks)
