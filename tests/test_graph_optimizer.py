from __future__ import annotations

import pytest

from src.graph_optimizer import GraphOptimizer


def test_graph_optimizer_collapses_duplicates() -> None:
    optimizer = GraphOptimizer()
    nodes = [
        {"id": "a", "worker": "WorkerA", "desc": "Collect data"},
        {"id": "b", "worker": "WorkerB", "desc": "Collect data"},
        {"id": "c", "worker": "WorkerC", "desc": "Summarize"},
    ]
    edges = [["a", "c"], ["b", "c"]]

    optimized_nodes, optimized_edges = optimizer.optimize(nodes, edges)

    assert len(optimized_nodes) == 2
    assert any(node["id"] == "c" for node in optimized_nodes)
    assert optimized_edges == [["a", "c"]]


def test_graph_optimizer_detects_cycles() -> None:
    optimizer = GraphOptimizer()
    nodes = [
        {"id": "a", "worker": "WorkerA", "desc": "Step A"},
        {"id": "b", "worker": "WorkerB", "desc": "Step B"},
    ]
    edges = [["a", "b"], ["b", "a"]]

    with pytest.raises(ValueError):
        optimizer.optimize(nodes, edges)
