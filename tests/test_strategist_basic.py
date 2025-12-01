from __future__ import annotations

from src.strategist.workflow_builder import Strategist


def test_strategist_creates_basic_workflow() -> None:
    strategist = Strategist()
    workflow = strategist.create_workflow("Summarize the latest AI news.")

    assert workflow.task == "Summarize the latest AI news."
    assert workflow.workflow_id
    assert len(workflow.execution_plan) == 1
    assert workflow.execution_plan[0].description == workflow.task