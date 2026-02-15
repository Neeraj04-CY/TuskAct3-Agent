from __future__ import annotations

import json
from pathlib import Path

from eikon_engine.learning.recorder import LearningRecorder


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_learning_recorder_marks_escalation(tmp_path: Path) -> None:
    mission_result_path = tmp_path / "mission_result.json"
    trace_path = tmp_path / "trace.json"
    _write_json(trace_path, {"id": "trace-esc", "failures": [], "skills_used": [], "subgoal_traces": []})
    _write_json(
        mission_result_path,
        {
            "mission_id": "mission_escalation",
            "status": "complete",
            "summary": {"escalation_state": {"used": True, "expanded_budget": {"max_steps": 40}}},
            "artifacts_path": str(tmp_path),
        },
    )

    recorder = LearningRecorder(output_dir=tmp_path / "learning_logs")
    recorder.record(
        mission_result_path=mission_result_path,
        trace_path=trace_path,
        mission_instruction="goal",
        runtime_error=None,
        force_outcome_failure=False,
        skill_summary=None,
    )

    payload = json.loads((tmp_path / "learning_logs" / "mission_escalation.json").read_text(encoding="utf-8"))
    assert payload["escalation_used"] is True
    assert payload["outcome"] == "escalation_success"
