"""Shared DAG utilities for strategist and workflow execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
from typing import Any, Dict, List, Optional

from src.telemetry import Telemetry
from src.tool_policy import ToolPolicy, ToolPolicyDecision


@dataclass(frozen=True)
class DAGNode:
    """Representation of a single task within a DAG."""

    node_id: str
    worker: str
    description: str


@dataclass
class DAG:
    """Minimalistic DAG structure with helper utilities."""

    nodes: Dict[str, DAGNode] = field(default_factory=dict)
    edges: Dict[str, List[str]] = field(default_factory=dict)

    def topological_levels(self) -> List[List[DAGNode]]:
        """Return nodes grouped by execution level following topological order."""

        indegree: Dict[str, int] = {node_id: 0 for node_id in self.nodes}
        for src, targets in self.edges.items():
            for dst in targets:
                if dst not in indegree:
                    raise ValueError(f"Edge references unknown node '{dst}'.")
                indegree[dst] += 1

        level: List[DAGNode] = [self.nodes[node_id] for node_id, deg in indegree.items() if deg == 0]
        if not level:
            raise ValueError("Plan has no starting nodes.")

        levels: List[List[DAGNode]] = []
        processed = 0
        indegree_copy = indegree.copy()

        while level:
            levels.append(level)
            processed += len(level)
            next_level: List[DAGNode] = []
            for node in level:
                for dst in self.edges.get(node.node_id, []):
                    indegree_copy[dst] -= 1
                    if indegree_copy[dst] == 0:
                        next_level.append(self.nodes[dst])
            level = next_level

        if processed != len(self.nodes):
            raise ValueError("Plan contains a cycle.")

        return levels


async def execute_dag(
    dag: DAG,
    worker_registry: Dict[str, Any],
    *,
    tool_policy: Optional[ToolPolicy] = None,
    telemetry: Optional[Telemetry] = None,
    iteration: int = 0,
) -> Dict[str, Any]:
    """Execute the DAG level-by-level using the provided workers."""

    results: Dict[str, Any] = {}
    session_store: Dict[str, Any] = {}
    try:
        for level in dag.topological_levels():
            nodes_to_run: List[DAGNode] = []
            tasks = []
            for node in level:
                decision: ToolPolicyDecision
                if tool_policy:
                    decision = tool_policy.evaluate(
                        {"worker": node.worker, "description": node.description, "id": node.node_id},
                        iteration=iteration,
                    )
                else:
                    decision = ToolPolicyDecision(True, None)

                if not decision.allowed:
                    results[node.node_id] = {"error": decision.reason or "Rejected by policy"}
                    if telemetry:
                        await telemetry.trace_event(
                            "error_detected",
                            {"node": node.node_id, "reason": decision.reason},
                        )
                    continue

                if telemetry:
                    await telemetry.trace_event(
                        "step_scheduled",
                        {"node": node.node_id, "worker": node.worker},
                    )
                tasks.append(_run_worker(node, worker_registry, results, session_store))
                nodes_to_run.append(node)

            level_outputs = await asyncio.gather(*tasks, return_exceptions=True)
            for node, output in zip(nodes_to_run, level_outputs):
                if isinstance(output, Exception):
                    results[node.node_id] = {"error": str(output)}
                    if telemetry:
                        await telemetry.trace_event(
                            "error_detected",
                            {"node": node.node_id, "reason": str(output)},
                        )
                else:
                    results[node.node_id] = output
                if telemetry:
                    await telemetry.trace_event(
                        "step_completed",
                        {"node": node.node_id, "worker": node.worker},
                    )
    finally:
        await _teardown_sessions(session_store)
    return results


async def _run_worker(
    node: DAGNode,
    worker_registry: Dict[str, Any],
    prior_results: Dict[str, Any],
    session_store: Dict[str, Any],
) -> Any:
    if node.worker not in worker_registry:
        raise ValueError(f"Worker '{node.worker}' is not registered.")

    worker_cls = worker_registry[node.worker]
    worker_instance = worker_cls()
    session_key = getattr(worker_cls, "session_key", None)
    kwargs: Dict[str, Any] = {}
    if session_key:
        kwargs["session"] = session_store.get(session_key)
        kwargs["reuse_session"] = True

    output = await worker_instance.run(node.description, dict(prior_results), **kwargs)
    if session_key and isinstance(output, dict):
        session_obj = output.pop("_session", None)
        if session_obj is not None:
            session_store[session_key] = session_obj
    return output


async def _teardown_sessions(session_store: Dict[str, Any]) -> None:
    if not session_store:
        return
    for session in session_store.values():
        if session is None:
            continue
        close = getattr(session, "close", None)
        if close is None:
            continue
        result = close()
        if inspect.isawaitable(result):
            await result
