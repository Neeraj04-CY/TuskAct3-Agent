from __future__ import annotations

from pathlib import Path

from run_offline_eval import run_offline_eval
from tests.replay.helpers import seed_autonomy_run


def test_offline_eval_generates_reports(tmp_path: Path) -> None:
    seed_autonomy_run(tmp_path)
    output_dir = tmp_path / "offline"
    result = run_offline_eval(artifact_root=tmp_path / "artifacts", output_dir=output_dir, save_hints=True)
    assert result["json_path"].exists()
    assert result["md_path"].exists()
    assert result["hints_path"] is not None
    assert (output_dir / "replay_summary.json").exists()
