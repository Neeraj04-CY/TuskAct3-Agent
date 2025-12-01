from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.common_types import WorkerStepResult, WorkflowObject, WorkflowStep
from src.worker.api_engine import ApiEngine, SimpleApiEngine
from src.worker.browser_engine import BrowserEngine, NoopBrowserEngine
from src.worker.code_engine import CodeEngine, NoopCodeEngine
from src.worker.file_engine import FileEngine, LocalFileEngine
from src.worker.logger import configure_logger


@dataclass
class WorkerConfig:
    max_retries_per_step: int = 3


class Worker:
    """
    Execution engine that executes the Strategist's WorkflowObject.

    Responsibilities:
    - Dispatch steps to appropriate engines (browser, API, code, file, etc.).
    - Handle retries and basic error reporting.
    - Log step-by-step progress.
    """

    def __init__(
        self,
        browser_engine: Optional[BrowserEngine] = None,
        api_engine: Optional[ApiEngine] = None,
        code_engine: Optional[CodeEngine] = None,
        file_engine: Optional[FileEngine] = None,
        config: Optional[WorkerConfig] = None,
        log_level: str = "INFO",
        log_dir: Optional[str] = None
    ) -> None:
        self._browser = browser_engine or NoopBrowserEngine()
        self._api = api_engine or SimpleApiEngine()
        self._code = code_engine or NoopCodeEngine()
        self._file = file_engine or LocalFileEngine()
        self._config = config or WorkerConfig()
        self._logger = configure_logger(log_level, log_dir)

    async def execute_workflow(self, workflow: WorkflowObject) -> List[WorkerStepResult]:
        """
        Execute all steps in the WorkflowObject sequentially (v1).
        Future versions may support parallelism and dynamic scheduling.
        """
        results: List[WorkerStepResult] = []
        self._logger.info("Starting workflow %s", workflow.workflow_id)

        for step in workflow.execution_plan:
            result = await self._execute_step(step)
            results.append(result)

        self._logger.info("Completed workflow %s", workflow.workflow_id)
        return results

    async def _execute_step(self, step: WorkflowStep) -> WorkerStepResult:
        """
        Execute a single step with retry logic.
        """
        self._logger.info("Executing step %s: %s", step.step_id, step.description)
        retries = 0

        while retries <= self._config.max_retries_per_step:
            try:
                outcome = await self._dispatch_step(step)
                return WorkerStepResult(
                    step=step.step_id,
                    status="success",
                    result=outcome,
                    error=None,
                    retry_count=retries,
                    artifacts=[],
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.exception("Error executing step %s: %s", step.step_id, exc)
                retries += 1
                if retries > self._config.max_retries_per_step:
                    return WorkerStepResult(
                        step=step.step_id,
                        status="failed",
                        result=None,
                        error=str(exc),
                        retry_count=retries - 1,
                        artifacts=[],
                    )

    async def _dispatch_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """
        Routes the step to the appropriate sub-engine based on tool/skill hints.
        v1 uses tool field as a simple router.
        """
        tool = step.tool

        if tool == "browser_engine":
            return await self._browser.run_action("generic", step.inputs)

        if tool == "api_engine":
            # Placeholder semantics: expect url + method in inputs
            method = step.inputs.get("method", "GET")
            url = step.inputs.get("url", "")
            return await self._api.call(method, url, params=step.inputs.get("params"))

        if tool == "code_engine":
            language = step.inputs.get("language", "python")
            code = step.inputs.get("code", "")
            return await self._code.run_code(language, code, context=step.inputs)

        # Default: treat as no-op description-only step.
        return {
            "description": step.description,
            "inputs": step.inputs,
            "note": "No specific tool assigned; step treated as informational."
        }