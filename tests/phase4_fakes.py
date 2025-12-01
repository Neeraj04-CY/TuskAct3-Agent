from __future__ import annotations

from typing import Any, Dict, List, Sequence

from eikon_engine.core.execution_utils import DAG, DAGNode, execute_dag


class StubMemoryManager:
    def __init__(self) -> None:
        self.saved: List[Dict[str, Any]] = []

    async def add_memory(self, event_type: str, title: str, text: str, context: Dict[str, Any]) -> str:
        entry = {
            "event_type": event_type,
            "title": title,
            "text": text,
            "context": context,
        }
        self.saved.append(entry)
        return f"mem-{len(self.saved)}"

    async def query_similar(self, text: str, k: int = 3) -> List[Dict[str, Any]]:  # noqa: D401
        return []


class FakeStrategist:
    def __init__(
        self,
        worker_registry: Dict[str, Any],
        plan_nodes: Sequence[Dict[str, str]],
        edges: Sequence[Sequence[str]],
        memory_manager: StubMemoryManager | None = None,
    ) -> None:
        self.worker_registry = worker_registry
        self.memory_manager = memory_manager
        self._plan_nodes = list(plan_nodes)
        self._edges = [list(edge) for edge in edges]

    async def plan(self, goal: str) -> Dict[str, Any]:
        nodes = {
            node["id"]: DAGNode(node_id=node["id"], worker=node["worker"], description=node["desc"])
            for node in self._plan_nodes
        }
        edge_map: Dict[str, List[str]] = {}
        for src, dst in self._edges:
            edge_map.setdefault(src, []).append(dst)
        dag = DAG(nodes=nodes, edges=edge_map)
        return {
            "goal": goal,
            "parsed_goal": {"raw": goal},
            "plan": {"nodes": list(self._plan_nodes), "edges": [list(edge) for edge in self._edges]},
            "dag": dag,
        }

    async def execute_dag(
        self,
        dag: DAG,
        worker_registry: Dict[str, Any],
        *,
        tool_policy=None,
        telemetry=None,
        iteration: int = 0,
    ) -> Dict[str, Any]:
        return await execute_dag(
            dag,
            worker_registry,
            tool_policy=tool_policy,
            telemetry=telemetry,
            iteration=iteration,
        )
