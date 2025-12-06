from __future__ import annotations

from pathlib import Path

from eikon_engine.replay.experience_replay import ExperienceReplayEngine
from tests.replay.helpers import seed_autonomy_run


def test_experience_replay_runs_batches(tmp_path: Path) -> None:
    seed_autonomy_run(tmp_path)
    engine = ExperienceReplayEngine(tmp_path / "artifacts")
    curriculum = engine.build_curriculum()
    result = engine.replay_curriculum(curriculum, output_dir=tmp_path / "replay")
    assert result["summary"]["states_processed"] >= 1
    replay_summary = (tmp_path / "replay" / "replay_summary.json")
    assert replay_summary.exists()
    memory_file = engine.save_memory_hints(tmp_path / "replay")
    assert memory_file.exists()
