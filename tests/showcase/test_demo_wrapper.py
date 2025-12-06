from __future__ import annotations

import json
from pathlib import Path

from run_demo import run_demo_once


def _demo_runner(goal: str, *, tmp_path: Path, **_: object) -> dict:
    run_dir = tmp_path / "autonomy" / "run_demo"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {"goal": goal, "completed": True, "reason": "ok"}
    result = {"steps": [], "run_context": {}}
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (run_dir / "stability_report.json").write_text(json.dumps({"metrics": {}}), encoding="utf-8")
    return {
        "summary": summary,
        "result": result,
        "run_dir": str(run_dir),
        "summary_path": str(run_dir / "summary.json"),
    }


def test_run_demo_once_produces_human_message(tmp_path: Path) -> None:
    def runner(goal: str, **kwargs: object) -> dict:
        return _demo_runner(goal, tmp_path=tmp_path, **kwargs)

    payload, message = run_demo_once("Test goal", demo_runner=runner)
    assert payload["summary"]["goal"] == "Test goal"
    assert message.startswith("PASS")
