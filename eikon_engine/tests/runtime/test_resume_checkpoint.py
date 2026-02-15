from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from eikon_engine.runtime.resume_checkpoint import ResumeCheckpoint

UTC = timezone.utc


def test_resume_checkpoint_round_trip(tmp_path: Path) -> None:
    checkpoint = ResumeCheckpoint(
        mission_id="mission_123",
        halted_subgoal_id="sg-1",
        halted_reason="judgment_refusal",
        page_url="https://example.com",
        page_intent="login_form",
        completed_subgoals=["sg-0"],
        pending_subgoals=["sg-1", "sg-2"],
        skills_used=["login_form_skill"],
        capability_state={"decisions": []},
        learning_bias_snapshot={"score": 0.5},
        trace_path="/tmp/trace.json",
        timestamp_utc=datetime.now(UTC).isoformat(),
        mission_instruction="Resume mission",
        artifacts_path="/tmp/mission_123",
    )

    path = checkpoint.save(tmp_path / "resume_checkpoint.json")
    loaded = ResumeCheckpoint.load(path)

    assert loaded == checkpoint
    assert loaded.artifacts_path == checkpoint.artifacts_path
