from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.common_types import WorkerStepResult, WorkflowObject, WorkflowSchedule
from src.memory.memory_manager import MemoryManager
from src.strategist.workflow_builder import Strategist
from src.worker.executor import Worker


@dataclass
class WorkflowEngineConfig:
    """
    Configuration for the Workflow Engine.
    """
    enable_memory: bool = False


class WorkflowEngine:
    """
    Orchestrates the entire lifecycle of a user request:

    - Accept user task
    - Call Strategist
    - Assemble Workflow
    - Call Worker
    - Handle retries (delegated to Worker)
    - Store memory (optional)
    - Generate final output
    """

    def __init__(
        self,
        strategist: Strategist,
        worker: Worker,
        memory_manager: MemoryManager | None = None,
        config: WorkflowEngineConfig | None = None
    ) -> None:
        self._strategist = strategist
        self._worker = worker
        self._memory = memory_manager
        self._config = config or WorkflowEngineConfig()

    def build_schedule(self, workflow: WorkflowObject) -> WorkflowSchedule:
        """
        Create a basic schedule from the WorkflowObject.

        v1: map execution_plan directly to execution; leave others empty.
        """
        return WorkflowSchedule(
            init={"workflow_id": workflow.workflow_id},
            pre_checks=[],
            execution=workflow.execution_plan,
            post_actions=[],
            memory_update=[],
        )

    async def run(self, user_input: str) -> Dict[str, Any]:
        """
        Public entrypoint for a single user task.

        Returns:
            Dict containing workflow metadata, step results, and a simple summary.
        """
        workflow = self._strategist.create_workflow(user_input)
        schedule = self.build_schedule(workflow)
        step_results: List[WorkerStepResult] = await self._worker.execute_workflow(workflow)

        # v1: no real memory integration; hook exists for future.
        if self._config.enable_memory and self._memory is not None:
            # TODO: derive embeddings and store success/failure.
            pass

        return {
            "workflow_id": workflow.workflow_id,
            "task": workflow.task,
            "results": [step_result.__dict__ for step_result in step_results],
        }