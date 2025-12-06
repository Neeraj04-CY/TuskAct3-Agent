from __future__ import annotations

import json

from eikon_engine.missions import cli


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
