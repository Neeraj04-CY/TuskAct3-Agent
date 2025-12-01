from __future__ import annotations

import argparse
import asyncio

from src.memory.memory_manager import MemoryManager
from src.strategist.workflow_builder import Strategist
from src.worker.executor import Worker
from src.workflow.engine import WorkflowEngine, WorkflowEngineConfig


async def _run_cli(task: str) -> None:
    strategist = Strategist()
    worker = Worker()
    memory_manager = MemoryManager()
    engine = WorkflowEngine(
        strategist=strategist,
        worker=worker,
        memory_manager=memory_manager,
        config=WorkflowEngineConfig(enable_memory=False),  # v1: disabled
    )

    result = await engine.run(task)
    print(f"Workflow ID: {result['workflow_id']}")
    print(f"Task: {result['task']}")
    print("Results:")
    for step in result["results"]:
        print(f"- {step['step']}: {step['status']} (retries={step['retry_count']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="EIKON ENGINE CLI")
    parser.add_argument("task", type=str, help="Natural language task to execute.")
    args = parser.parse_args()

    asyncio.run(_run_cli(args.task))


if __name__ == "__main__":
    main()