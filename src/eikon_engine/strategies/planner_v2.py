"""Simple dependency-aware planner that produces a DAG-style plan."""

from __future__ import annotations

import re
from typing import Any, Dict, List

_WORKER_POOL = ["WorkerA", "WorkerB", "WorkerC", "WorkerD"]
FILE_URL_RE = re.compile(r"file://[^\s]+", re.IGNORECASE)


def _extract_safe_urls(text: str) -> List[str]:
    """Return all file:// URLs exactly as they appear, without truncation."""

    return FILE_URL_RE.findall(text)


def plan(goal: str) -> Dict[str, Any]:
    """Create a lightweight dependency graph for the provided goal."""
    normalized_goal = goal.strip()
    if not normalized_goal:
        raise ValueError("Goal cannot be empty.")

    task_descriptions = _generate_task_descriptions(normalized_goal)
    nodes = _build_nodes(task_descriptions)
    edges = _build_edges(nodes)

    return {"goal": normalized_goal, "nodes": nodes, "edges": edges}


def _generate_task_descriptions(goal: str) -> List[str]:
    """Craft human-readable task descriptions derived from the goal."""

    goal_text = goal
    file_urls = _extract_safe_urls(goal_text)
    frozen_map: Dict[str, str] = {}
    for idx, url in enumerate(file_urls):
        placeholder = f"__FILE_URL_{idx}__"
        frozen_map[placeholder] = url
        goal_text = goal_text.replace(url, placeholder)

    target_count = _determine_task_count(goal)
    fragments = [frag.strip() for frag in goal_text.replace("?", ".").split(".") if frag.strip()]

    descriptions: List[str] = []
    for idx in range(target_count):
        source = fragments[idx % len(fragments)] if fragments else goal_text
        descriptions.append(f"Task {idx + 1}: {source}")

    if frozen_map:
        for placeholder, url in frozen_map.items():
            descriptions = [desc.replace(placeholder, url) for desc in descriptions]

    return descriptions


def _determine_task_count(goal: str) -> int:
    """Return task count between 2 and 4 based on goal verbosity."""

    word_count = len(goal.split())
    return max(2, min(4, (word_count // 15) + 1))


def _build_nodes(descriptions: List[str]) -> List[Dict[str, str]]:
    """Map each description onto a worker following a round-robin pattern."""

    nodes: List[Dict[str, str]] = []
    for idx, desc in enumerate(descriptions, start=1):
        worker = _WORKER_POOL[(idx - 1) % len(_WORKER_POOL)]
        nodes.append({
            "id": f"task{idx}",
            "worker": worker,
            "desc": desc,
        })
    return nodes


def _build_edges(nodes: List[Dict[str, str]]) -> List[List[str]]:
    """Link all tasks to the final task, producing a simple DAG."""

    if len(nodes) < 2:
        return []
    final_task_id = nodes[-1]["id"]
    return [[node["id"], final_task_id] for node in nodes[:-1]]
