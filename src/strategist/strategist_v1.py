"""Strategist v1 upgraded with DAG-aware planning and execution."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Union

from eikon_engine.core import DAG, DAGNode, execute_dag as execute_dag_helper
from eikon_engine.strategies import plan as planner_v2
from src.completion import extract_completion_metadata
from src.graph_optimizer import GraphOptimizer
from src.memory.memory_manager import MemoryManager
from src.telemetry import Telemetry
from src.tool_policy import ToolPolicy

LOGGER = logging.getLogger(__name__)

FILE_URL_RE = re.compile(r"file://[^\s]+", re.IGNORECASE)


def _extract_safe_urls(text: str) -> List[str]:
    """Return every file:// URL exactly as authored."""

    return FILE_URL_RE.findall(text or "")


def _freeze_file_urls(goal_text: str) -> tuple[str, Dict[str, str]]:
    """Replace file URLs with placeholders to avoid planner truncation."""

    frozen: Dict[str, str] = {}
    if not goal_text:
        return goal_text, frozen
    for idx, url in enumerate(_extract_safe_urls(goal_text)):
        placeholder = f"__FILE_URL_{idx}__"
        frozen[placeholder] = url
        goal_text = goal_text.replace(url, placeholder)
    return goal_text, frozen


def _restore_placeholders(text: str, frozen_map: Dict[str, str]) -> str:
    restored = text
    for placeholder, url in frozen_map.items():
        if placeholder in restored:
            restored = restored.replace(placeholder, url)
    return restored


def _restore_placeholders_in_plan(plan_dict: Dict[str, Any], frozen_map: Dict[str, str]) -> None:
    """Mutate plan entries so placeholders are swapped back to full URLs."""

    if not frozen_map:
        return
    for field in ("goal", "description", "summary", "text"):
        value = plan_dict.get(field)
        if isinstance(value, str):
            plan_dict[field] = _restore_placeholders(value, frozen_map)
    nodes = plan_dict.get("nodes")
    if not isinstance(nodes, list):
        return
    for node in nodes:
        if not isinstance(node, dict):
            continue
        for key in ("desc", "description", "summary"):
            value = node.get(key)
            if isinstance(value, str):
                node[key] = _restore_placeholders(value, frozen_map)


class GoalWorker(Protocol):
    """Protocol describing the worker interface used by Strategist."""

    async def run(self, description: str, prev_results: Dict[str, Any]) -> Any:
        """Execute work for a single task."""


@dataclass
class Assignment:
    """Legacy linear assignment metadata for backward compatibility."""

    step: str
    worker: str


class Strategist:
    """Dependency-aware Strategist with backward-compatible APIs."""

    def __init__(
        self,
        worker_registry: Dict[str, Any],
        memory_manager: MemoryManager | None = None,
        *,
        graph_optimizer: GraphOptimizer | None = None,
        tool_policy: ToolPolicy | None = None,
        telemetry: Telemetry | None = None,
    ) -> None:
        self.worker_registry = worker_registry
        self.memory_manager = memory_manager
        self.graph_optimizer = graph_optimizer or GraphOptimizer()
        self.tool_policy = tool_policy or ToolPolicy(allowed_workers=worker_registry.keys())
        self.telemetry = telemetry or Telemetry(memory_manager)

    async def __call__(self, goal: str) -> Dict[str, Any]:
        """Allow the strategist instance to be invoked directly."""

        return await self._run_with_dag(goal)

    async def run(self, goal: str) -> Dict[str, Any]:
        """Public entrypoint matching previous versions."""

        return await self._run_with_dag(goal)

    async def plan(self, goal: str) -> Dict[str, Any]:
        """Create an optimized plan and DAG for downstream orchestrators."""

        parsed_goal = self.parse_goal(goal)
        parsed_goal = await self._attach_related_memories(parsed_goal)
        frozen_map: Dict[str, str] = {}
        goal_for_planner = goal
        if "file://" in goal.lower():
            goal_for_planner, frozen_map = _freeze_file_urls(goal)
        plan_dict = planner_v2(goal_for_planner)
        if frozen_map:
            _restore_placeholders_in_plan(plan_dict, frozen_map)
        nodes = plan_dict.get("nodes", [])
        edges = plan_dict.get("edges", [])
        optimized_nodes, optimized_edges = self.graph_optimizer.optimize(nodes, edges)
        plan_dict = {
            **plan_dict,
            "nodes": optimized_nodes,
            "edges": optimized_edges,
        }
        dag = self.build_dag(plan_dict)
        plan_dict = {
            **plan_dict,
            "nodes": [
                {
                    "id": node.node_id,
                    "worker": node.worker,
                    "desc": node.description,
                }
                for node in dag.nodes.values()
            ],
        }
        if self.telemetry:
            await self.telemetry.trace_event(
                "plan_created",
                {"goal": goal, "nodes": len(optimized_nodes)},
            )
        return {
            "goal": goal,
            "parsed_goal": parsed_goal,
            "plan": plan_dict,
            "dag": dag,
        }

    async def _run_with_dag(self, goal: str) -> Dict[str, Any]:
        plan_bundle = await self.plan(goal)
        dag = plan_bundle["dag"]
        plan_dict = plan_bundle["plan"]
        parsed_goal = plan_bundle["parsed_goal"]
        dag_results = await self.execute_dag(
            dag,
            self.worker_registry,
            tool_policy=self.tool_policy,
            telemetry=self.telemetry,
        )
        completion_meta = extract_completion_metadata(dag_results)
        memory_written, memory_error = await self._write_memory_post_execution(goal, plan_dict, dag_results)
        final_payload: Dict[str, Any] = {
            "goal": goal,
            "parsed_goal": parsed_goal,
            "plan": plan_dict,
            "results": dag_results,
        }
        if completion_meta:
            final_payload["completion"] = completion_meta
        if "related_memories" in parsed_goal:
            final_payload["related_memories"] = parsed_goal["related_memories"]
        final_payload["memory_written"] = memory_written
        if memory_error:
            final_payload["memory_write_error"] = memory_error
        return final_payload

    def parse_goal(self, goal: str) -> Dict[str, Any]:
        """Retain minimal parsing used by legacy pipeline."""

        return {"raw": goal, "length": len(goal)}

    async def _attach_related_memories(self, parsed_goal: Dict[str, Any]) -> Dict[str, Any]:
        """Optionally annotate the parsed goal with related memories."""

        if not self.memory_manager:
            return parsed_goal

        query_text = parsed_goal.get("raw", "").strip()
        if not query_text:
            return parsed_goal

        enriched_goal = dict(parsed_goal)
        try:
            matches = await self.memory_manager.query_similar(query_text, k=3)
            enriched_goal["related_memories"] = [
                {
                    "id": match.get("id"),
                    "score": float(match.get("score", 0.0)),
                    "metadata": match.get("metadata", {}),
                }
                for match in matches
            ]
        except Exception as exc:  # noqa: BLE001
            enriched_goal["memory_error"] = str(exc)
        return enriched_goal

    def plan_workflow(self, parsed_goal: Dict[str, Any]) -> List[str]:
        """Legacy linear plan used by older components/tests."""

        raw_goal = parsed_goal["raw"]
        return [
            f"Understand the request: {raw_goal}",
            "Generate a structured response based on understanding.",
        ]

    def assign_workers(self, steps: List[str]) -> List[Assignment]:
        """Legacy round-robin assignment helper."""

        if not self.worker_registry:
            raise ValueError("No workers available for assignments.")

        worker_names = list(self.worker_registry.keys())
        assignments: List[Assignment] = []
        for index, step in enumerate(steps):
            worker_name = worker_names[index % len(worker_names)]
            assignments.append(Assignment(step=step, worker=worker_name))
        return assignments

    async def execute_workflow(self, assignments: List[Union[Assignment, Dict[str, str]]]) -> Dict[str, Any]:
        """Legacy sequential execution helper."""

        results: Dict[str, Any] = {}
        for assignment in assignments:
            if isinstance(assignment, Assignment):
                worker_name = assignment.worker
                description = assignment.step
            else:
                worker_name = assignment["worker"]
                description = assignment["step"]

            worker_cls = self.worker_registry[worker_name]
            worker_instance = worker_cls()
            prev_results = dict(results)
            try:
                output = await worker_instance.run(description, prev_results)
            except Exception as exc:  # noqa: BLE001
                output = {"error": str(exc)}
            results[description] = output
        return results

    def produce_output(
        self,
        parsed_goal: Dict[str, Any],
        plan: List[str],
        assignments: List[Assignment],
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Legacy helper left intact for compatibility with older callers."""

        return {
            "parsed_goal": parsed_goal,
            "plan": plan,
            "assignments": [assignment.__dict__ for assignment in assignments],
            "results": results,
        }

    def build_dag(self, plan: Dict[str, Any]) -> DAG:
        """Convert planner output into a DAG structure."""

        if not self.worker_registry:
            raise ValueError("Worker registry cannot be empty.")

        node_entries = plan.get("nodes", [])
        if not node_entries:
            raise ValueError("Plan must include at least one node.")

        worker_names = list(self.worker_registry.keys())
        nodes: Dict[str, DAGNode] = {}
        for index, node in enumerate(node_entries):
            node_id = node["id"]
            worker_name = node.get("worker", worker_names[index % len(worker_names)])
            if worker_name not in self.worker_registry:
                worker_name = worker_names[index % len(worker_names)]
            nodes[node_id] = DAGNode(
                node_id=node_id,
                worker=worker_name,
                description=node["desc"],
            )
        edges: Dict[str, List[str]] = {}
        for src, dst in plan.get("edges", []):
            edges.setdefault(src, []).append(dst)
        return DAG(nodes=nodes, edges=edges)

    async def execute_dag(
        self,
        dag: DAG,
        worker_registry: Dict[str, Any],
        *,
        tool_policy: ToolPolicy | None = None,
        telemetry: Telemetry | None = None,
        iteration: int = 0,
    ) -> Dict[str, Any]:
        """Execute the DAG by delegating to the shared helper."""

        return await execute_dag_helper(
            dag,
            worker_registry,
            tool_policy=tool_policy or self.tool_policy,
            telemetry=telemetry or self.telemetry,
            iteration=iteration,
        )

    async def _write_memory_post_execution(
        self,
        goal: str,
        plan: Dict[str, Any],
        results: Dict[str, Any],
    ) -> tuple[Optional[str], Optional[str]]:
        """Persist a summary of the execution when memory is available."""

        if not self.memory_manager:
            return None, None

        context = {
            "plan_nodes": list(plan.keys()),
            "worker_count": len(plan),
        }
        summary = f"Goal: {goal}\nPlan: {plan}\nResults: {results}"
        try:
            memory_id = await self.memory_manager.add_memory(
                event_type="workflow_execution",
                title=f"Execution for: {goal[:40]}",
                text=summary,
                context=context,
            )
            LOGGER.info("Stored workflow execution memory id=%s", memory_id)
            if self.telemetry:
                await self.telemetry.trace_event(
                    "memory_written",
                    {"memory_id": memory_id, "goal": goal[:60]},
                )
            return memory_id, None
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to store workflow execution memory: %s", exc)
            if self.telemetry:
                await self.telemetry.trace_event(
                    "error_detected",
                    {"goal": goal[:60], "reason": str(exc)},
                )
            return None, str(exc)
