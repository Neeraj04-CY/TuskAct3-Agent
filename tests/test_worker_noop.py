from __future__ import annotations

import asyncio

from src.common_types import WorkflowObject, WorkflowStep
from src.worker.executor import Worker


def test_worker_executes_noop_step() -> None:
    worker = Worker()
    workflow = WorkflowObject(
        workflow_id="test",
        task="noop",
        subtasks=["noop"],
        tools_needed=[],
        skills_to_load=[],
        memory_references=[],
        constraints=[],
        risk_analysis=[],
        success_criteria=[],
        estimated_time="0",
        execution_plan=[
            WorkflowStep(
                step_id="s1",
                description="Do nothing",
                tool=None,
                skill=None,
                inputs={},
                depends_on=[],
            )
        ],
    )

    results = asyncio.run(worker.execute_workflow(workflow))
    assert len(results) == 1
    assert results[0].status == "success"