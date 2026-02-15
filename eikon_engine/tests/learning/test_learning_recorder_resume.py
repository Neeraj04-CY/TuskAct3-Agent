from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from eikon_engine.learning.recorder import LearningRecorder

UTC = timezone.utc


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_learning_recorder_marks_resumed_success(tmp_path: Path) -> None:
    mission_id = "mission_resumed"
    mission_result_path = tmp_path / "mission_result.json"
    trace_path = tmp_path / "trace.json"
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "dummy.txt").write_text("ok", encoding="utf-8")

    _write_json(trace_path, {"id": "trace-1", "failures": [], "skills_used": [], "subgoal_traces": []})
    _write_json(
        mission_result_path,
        {
            "mission_id": mission_id,
            "status": "complete",
            "summary": {"resumed_from_checkpoint": "artifacts/mission_resumed/resume_checkpoint.json"},
            "artifacts_path": str(artifacts_dir),
        },
    )

    recorder = LearningRecorder(output_dir=tmp_path / "learning_logs")
    recorder.record(
        mission_result_path=mission_result_path,
        trace_path=trace_path,
        mission_instruction="resume test",
        runtime_error=None,
        force_outcome_failure=False,
        skill_summary=None,
    )

    record_path = tmp_path / "learning_logs" / f"{mission_id}.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["resumed"] is True
    assert payload["resume_source"].endswith("resume_checkpoint.json")
    assert payload["outcome"] == "resumed_success"


def test_learning_recorder_marks_resumed_failure(tmp_path: Path) -> None:
    mission_id = "mission_resumed_fail"
    mission_result_path = tmp_path / "mission_result.json"
    trace_path = tmp_path / "trace.json"
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "dummy.txt").write_text("ok", encoding="utf-8")

    _write_json(trace_path, {"id": "trace-2", "failures": [], "skills_used": [], "subgoal_traces": []})
    _write_json(
        mission_result_path,
        {
            "mission_id": mission_id,
            "status": "failed",
            "summary": {"resumed_from_checkpoint": "artifacts/mission_resumed/resume_checkpoint.json"},
            "artifacts_path": str(artifacts_dir),
        },
    )

    recorder = LearningRecorder(output_dir=tmp_path / "learning_logs")
    recorder.record(
        mission_result_path=mission_result_path,
        trace_path=trace_path,
        mission_instruction="resume failure",
        runtime_error=RuntimeError("boom"),
        force_outcome_failure=False,
        skill_summary=None,
    )

    record_path = tmp_path / "learning_logs" / f"{mission_id}.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["resumed"] is True
    assert payload["outcome"] == "resumed_failure"
