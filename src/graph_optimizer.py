from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Tuple


class GraphOptimizer:
    """Performs lightweight DAG optimizations before execution."""

    def optimize(
        self,
        nodes: Iterable[Dict[str, str]],
        edges: Iterable[Iterable[str]],
    ) -> Tuple[List[Dict[str, str]], List[List[str]]]:
        unique_nodes: List[Dict[str, str]] = []
        seen_descriptions: set[str] = set()

        for node in nodes:
            node_id = node["id"]
            desc_key = node.get("desc", "").strip().lower()
            if desc_key and desc_key in seen_descriptions:
                continue
            seen_descriptions.add(desc_key)
            unique_nodes.append(node)

        node_ids = {node["id"] for node in unique_nodes}
        filtered_edges: List[List[str]] = [
            [src, dst]
            for src, dst in edges
            if src in node_ids and dst in node_ids
        ]

        ordered_nodes = self._topological_order(unique_nodes, filtered_edges)
        return ordered_nodes, filtered_edges

    def _topological_order(self, nodes: List[Dict[str, str]], edges: List[List[str]]) -> List[Dict[str, str]]:
        adjacency: Dict[str, List[str]] = {node["id"]: [] for node in nodes}
        indegree: Dict[str, int] = {node_id: 0 for node_id in adjacency}
        for src, dst in edges:
            adjacency.setdefault(src, []).append(dst)
            indegree[dst] = indegree.get(dst, 0) + 1

        queue: deque[str] = deque([node_id for node_id, deg in indegree.items() if deg == 0])
        ordered: List[Dict[str, str]] = []
        visited = 0
        while queue:
            current = queue.popleft()
            visited += 1
            ordered.append(next(node for node in nodes if node["id"] == current))
            for neighbor in adjacency.get(current, []):
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        if visited != len(nodes):
            raise ValueError("Plan contains a cycle; cannot optimize")
        return ordered
