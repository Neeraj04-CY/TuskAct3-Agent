from __future__ import annotations

import json
from datetime import datetime, timezone

from eikon_engine.missions import cli
from eikon_engine.missions.mission_schema import MissionResult, MissionSubgoalResult, mission_id

UTC = timezone.utc


def test_cli_main_uses_test_executor(monkeypatch, tmp_path):
    artifact_root = tmp_path / "cli_artifacts"
    hook = "eikon_engine.tests.missions.cli_stub:build_executor"
    monkeypatch.setenv(cli._TEST_EXECUTOR_ENV, hook)
    argv = [
        "--mission",
        "Check the latest market close",
        "--timeout",
        "60",
        "--max-retries",
        "1",
        "--constraints",
        json.dumps({"symbol": "MSFT"}),
        "--artifacts-dir",
        str(artifact_root),
    ]
    exit_code = cli.main(argv)
    assert exit_code == 0
    mission_dirs = list(artifact_root.iterdir())
    assert mission_dirs, "CLI should create at least one mission directory"
    result_path = mission_dirs[0] / "mission_result.json"
    assert result_path.exists()
    mission_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert mission_result["status"] == "complete"
    assert mission_result["summary"]["reason"] == "stub"


def test_cli_main_passes_guardrail_flags(monkeypatch, tmp_path):
    captured = {}

    def fake_run(spec, executor=None, resume_from=None):  # noqa: D401
        captured["spec"] = spec
        now = datetime.now(UTC)
        subgoal = MissionSubgoalResult(
            subgoal_id="sg",
            description="stub",
            status="complete",
            attempts=1,
            started_at=now,
            ended_at=now,
        )
        return MissionResult(
            mission_id=mission_id(),
            status="complete",
            start_ts=now,
            end_ts=now,
            subgoal_results=[subgoal],
            summary={"reason": "ok"},
            artifacts_path=str(tmp_path),
        )

    monkeypatch.setattr(cli, "run_mission_sync", fake_run)
    monkeypatch.setattr(cli, "_resolve_executor", lambda *args, **kwargs: object())
    argv = [
        "--mission",
        "Guard",
        "--autonomy-budget",
        json.dumps({"max_steps": 5}),
        "--safety-contract",
        json.dumps({"blocked_actions": ["download_file"]}),
        "--ask-on-uncertainty",
    ]

    exit_code = cli.main(argv)

    assert exit_code == 0
    spec = captured["spec"]
    assert spec.autonomy_budget["max_steps"] == 5
    assert spec.safety_contract["blocked_actions"] == ["download_file"]
    assert spec.ask_on_uncertainty is True


def test_cli_main_resume_passes_checkpoint(monkeypatch, tmp_path):
    captured = {}

    def fake_run(spec, executor=None, resume_from=None):  # noqa: D401
        captured["spec"] = spec
        captured["resume_from"] = resume_from
        now = datetime.now(UTC)
        subgoal = MissionSubgoalResult(
            subgoal_id="sg",
            description="resume stub",
            status="complete",
            attempts=1,
            started_at=now,
            ended_at=now,
        )
        return MissionResult(
            mission_id=mission_id(),
            status="complete",
            start_ts=now,
            end_ts=now,
            subgoal_results=[subgoal],
            summary={"reason": "stub"},
            artifacts_path=str(tmp_path),
        )

    monkeypatch.setattr(cli, "run_mission_sync", fake_run)
    monkeypatch.setattr(cli, "_resolve_executor", lambda *args, **kwargs: object())
    argv = ["--resume", "mission_123", "--artifacts-dir", str(tmp_path)]

    exit_code = cli.main(argv)

    assert exit_code == 0
    assert captured["resume_from"] == "mission_123"
    assert captured["spec"].instruction == "resume-mission"
