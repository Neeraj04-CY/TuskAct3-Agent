from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from eikon_engine.learning.diff import (
    build_learning_summary,
    build_skill_diff_report,
    emit_learning_artifacts,
    write_learning_diff_artifact,
    write_learning_summary,
)
from eikon_engine.learning.signals import SkillSignal

UTC = timezone.utc


def _signal(*, skill: str, mission_type: str, attempts: int, successes: int, steps_saved: int, confidence: float, mission_id: str, timestamp: datetime) -> SkillSignal:
    return SkillSignal(
        skill_name=skill,
        mission_type=mission_type,
        attempts=attempts,
        successes=successes,
        total_steps_saved=steps_saved,
        last_mission_id=mission_id,
        last_timestamp=timestamp,
        confidence_samples=max(1, attempts),
        confidence_mean=confidence,
    )


def test_learning_diff_emitted_on_change(tmp_path: Path) -> None:
    before = [
        _signal(
            skill="login_form_skill",
            mission_type="login",
            attempts=1,
            successes=1,
            steps_saved=2,
            confidence=0.72,
            mission_id="m-1",
            timestamp=datetime(2026, 1, 13, 12, 0, tzinfo=UTC),
        )
    ]
    after = [
        _signal(
            skill="login_form_skill",
            mission_type="login",
            attempts=3,
            successes=3,
            steps_saved=8,
            confidence=0.9,
            mission_id="m-2",
            timestamp=datetime(2026, 1, 13, 13, 0, tzinfo=UTC),
        )
    ]

    report = build_skill_diff_report("mission-123", before_signals=before, after_signals=after)
    diff_path = write_learning_diff_artifact(tmp_path, report)
    payload = json.loads(diff_path.read_text(encoding="utf-8"))

    assert payload["skill_diffs"], "expected diff entries when learning signals change"
    entry = payload["skill_diffs"][0]
    assert entry["skill"] == "login_form_skill"
    assert "priority" in entry["before"] and "priority" in entry["after"]
    assert "success_rate" in entry["reason"]


def test_learning_diff_empty_when_no_change(tmp_path: Path) -> None:
    timestamp = datetime(2026, 1, 13, 12, 0, tzinfo=UTC)
    baseline = [
        _signal(
            skill="login_form_skill",
            mission_type="login",
            attempts=2,
            successes=2,
            steps_saved=4,
            confidence=0.85,
            mission_id="m-1",
            timestamp=timestamp,
        )
    ]

    report = build_skill_diff_report("mission-456", before_signals=baseline, after_signals=baseline)
    diff_path = write_learning_diff_artifact(tmp_path, report)
    payload = json.loads(diff_path.read_text(encoding="utf-8"))

    assert payload["skill_diffs"] == []


def test_learning_summary_written(tmp_path: Path) -> None:
    before = [
        _signal(
            skill="login_form_skill",
            mission_type="login",
            attempts=1,
            successes=1,
            steps_saved=2,
            confidence=0.7,
            mission_id="m-a",
            timestamp=datetime(2026, 1, 13, 11, 30, tzinfo=UTC),
        )
    ]
    after = [
        _signal(
            skill="login_form_skill",
            mission_type="login",
            attempts=2,
            successes=2,
            steps_saved=5,
            confidence=0.88,
            mission_id="m-b",
            timestamp=datetime(2026, 1, 13, 13, 30, tzinfo=UTC),
        )
    ]

    report = build_skill_diff_report("mission-789", before_signals=before, after_signals=after)
    summary = build_learning_summary(report)
    summary_path = write_learning_summary(tmp_path, summary)

    text = summary_path.read_text(encoding="utf-8").strip()
    assert text.startswith("Learning applied successfully.")
    assert "priority" in text or "success_rate" in text


def test_replay_unchanged_by_learning(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    trace_payload = {"id": "trace-demo", "status": "complete"}
    trace_path.write_text(json.dumps(trace_payload), encoding="utf-8")
    baseline_trace = trace_path.read_text(encoding="utf-8")

    before = [
        _signal(
            skill="login_form_skill",
            mission_type="login",
            attempts=1,
            successes=1,
            steps_saved=3,
            confidence=0.8,
            mission_id="m-before",
            timestamp=datetime(2026, 1, 13, 10, 0, tzinfo=UTC),
        )
    ]
    after = [
        _signal(
            skill="login_form_skill",
            mission_type="login",
            attempts=2,
            successes=2,
            steps_saved=6,
            confidence=0.9,
            mission_id="m-after",
            timestamp=datetime(2026, 1, 13, 11, 0, tzinfo=UTC),
        )
    ]

    emit_learning_artifacts(
        mission_id="mission-demo",
        mission_dir=tmp_path,
        before_signals=before,
        after_signals=after,
    )

    assert trace_path.read_text(encoding="utf-8") == baseline_trace
