from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict

from eikon_engine.core.execution_utils import DAG, DAGNode, execute_dag


class _FakeSession:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@dataclass
class _CallRecord:
    description: str
    session_before: Any


class _FakeBrowserWorker:
    session_key = "browser"
    calls: list[_CallRecord] = []
    created_session: _FakeSession | None = None

    async def run(
        self,
        description: str,
        prev_results: Dict[str, Any],
        *,
        session: _FakeSession | None = None,
        reuse_session: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:
        del prev_results
        _FakeBrowserWorker.calls.append(_CallRecord(description=description, session_before=session))
        if reuse_session:
            if session is None:
                session = _FakeSession()
                _FakeBrowserWorker.created_session = session
            return {"result": description, "_session": session}
        return {"result": description}


def test_execute_dag_reuses_browser_session_and_tears_down() -> None:
    _FakeBrowserWorker.calls = []
    _FakeBrowserWorker.created_session = None

    dag = DAG(
        nodes={
            "task1": DAGNode(node_id="task1", worker="BrowserWorker", description="Navigate to file"),
            "task2": DAGNode(node_id="task2", worker="BrowserWorker", description="Fill form"),
        },
        edges={"task1": ["task2"]},
    )

    worker_registry = {"BrowserWorker": _FakeBrowserWorker}
    results = asyncio.run(execute_dag(dag, worker_registry))

    assert results["task1"]["result"] == "Navigate to file"
    assert results["task2"]["result"] == "Fill form"

    assert len(_FakeBrowserWorker.calls) == 2
    assert _FakeBrowserWorker.calls[0].session_before is None
    assert _FakeBrowserWorker.calls[1].session_before is _FakeBrowserWorker.created_session
    assert _FakeBrowserWorker.created_session is not None
    assert _FakeBrowserWorker.created_session.closed is True
