from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

from eikon_engine.missions.mission_executor import MissionExecutor
from eikon_engine.missions.mission_schema import MissionSpec, MissionSubgoal
from eikon_engine.trace.recorder import ExecutionTraceRecorder
from eikon_engine.trace.reader import read_trace

UTC = timezone.utc


class _TraceAwareWorker:
    def __init__(self) -> None:
        self.logger = None
        self.demo_mode = False
        self._trace_recorder: ExecutionTraceRecorder | None = None
        self._trace_handle: str | None = None

    async def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            "completion": {"complete": True, "reason": "ok"},
            "error": None,
            "artifacts": {},
        }
        if self._trace_recorder and self._trace_handle:
            started = datetime.now(UTC)
            self._trace_recorder.record_action(
                self._trace_handle,
                action_type="noop",
                selector=None,
                target=None,
                input_data=None,
                status="ok",
                started_at=started,
                ended_at=started,
                duration_ms=0.0,
                metadata={"test": True, "payload": payload},
            )
        return result

    async def shutdown(self) -> None:
        return None

    def set_mission_context(self, **_: object) -> None:
        return None

    def set_trace_context(self, *, trace_recorder: ExecutionTraceRecorder | None, trace_handle: str | None) -> None:
        self._trace_recorder = trace_recorder
        self._trace_handle = trace_handle

    def clear_trace_context(self) -> None:
        self._trace_handle = None


def _build_executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MissionExecutor:
    trace_dir = tmp_path / "traces"
    worker = _TraceAwareWorker()
    monkeypatch.setattr(MissionExecutor, "_build_worker", lambda self, spec: worker)
    return MissionExecutor(
        settings={"planner": {}},
        artifacts_root=tmp_path,
        trace_recorder_factory=lambda: ExecutionTraceRecorder(storage_dir=trace_dir),
    )


def _trace_files(trace_dir: Path) -> tuple[Path, Path]:
    trace_file = next(trace_dir.glob("trace_*/trace.json"))
    summary_file = trace_file.with_name("trace_summary.txt")
    return trace_file, summary_file


@pytest.mark.asyncio
async def test_trace_integrity_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = MissionSpec(instruction="Mock mission", execute=False)
    trace_dir = tmp_path / "traces"
    subgoal = MissionSubgoal(
        id="sg-noop",
        description="Bootstrap",
        planner_metadata={"bootstrap_actions": [{"action": "noop"}]},
    )
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda _: [subgoal])

    executor = _build_executor(tmp_path, monkeypatch)
    result = await executor.run_mission(spec)

    assert result.status == "complete"
    trace_file, summary_file = _trace_files(trace_dir)
    trace = read_trace(trace_file)

    assert trace.trace_version == "v3.1"
    assert len(trace.subgoal_traces) >= 1
    assert trace.subgoal_traces[0].actions_taken, "expected at least one action trace"
    assert trace.subgoal_traces[0].actions_taken[0].status == "ok"
    assert summary_file.exists()
