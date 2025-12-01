"""Workflow DAG executor mirroring Strategist semantics."""

from __future__ import annotations

from typing import Any, Dict

from eikon_engine.core import DAG, execute_dag as execute_dag_helper


async def execute_graph(dag: DAG, worker_registry: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the provided DAG using the shared execution helper."""

    return await execute_dag_helper(dag, worker_registry)
