from __future__ import annotations

import json
from pathlib import Path

import pytest

from eikon_engine.missions.mission_executor import MissionExecutor
from eikon_engine.missions.mission_schema import MissionSpec, MissionSubgoal


@pytest.mark.asyncio
async def test_mission_executor_happy_path_async(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = MissionSpec(instruction="Collect a screenshot", execute=False)
    subgoals = [MissionSubgoal(id="sg1", description="Open page", planner_metadata={})]
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", lambda _spec: subgoals)

    async def fake_run_pipeline(self, **_: object) -> dict:  # noqa: ARG001 - signature matches monkeypatching needs
        return {
            "completion": {"complete": True, "reason": "ok"},
            "artifacts": {"base_dir": str(tmp_path / "subgoal")},
        }

    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", fake_run_pipeline)
    executor = MissionExecutor(
        settings={"planner": {}, "logging": {"artifact_root": str(tmp_path)}},
        artifacts_root=tmp_path,
    )
    result = await executor.run_mission(spec)
    assert result.status == "complete"
    mission_file = Path(result.artifacts_path) / "mission_result.json"
    assert mission_file.exists()
    payload = json.loads(mission_file.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"


@pytest.mark.asyncio
async def test_mission_executor_runs_bootstrap_navigation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bootstrap_subgoal = MissionSubgoal(
        id="sg0",
        description='{"action": "navigate", "url": "https://acme.test/login"}',
        planner_metadata={
            "bootstrap_actions": [{"action": "navigate", "url": "https://acme.test/login"}],
        },
    )
    followup_subgoal = MissionSubgoal(id="sg1", description="Check login form", planner_metadata={})
    retry_subgoal = MissionSubgoal(id="sg2", description="Form: retry", planner_metadata={})
    monkeypatch.setattr(
        "eikon_engine.missions.mission_executor.plan_mission",
        lambda _spec: [bootstrap_subgoal, followup_subgoal, retry_subgoal],
    )

    class FakeWorker:
        def __init__(self) -> None:
            self.logger = None
            self.payloads = []
            self.shutdown_called = False

        async def execute(self, payload):  # type: ignore[override]
            self.payloads.append(payload)
            screenshot_path: str
            if self.logger:
                screenshot = self.logger.save_screenshot(b"secure", step_index=1, name="secure_area.png")
                screenshot_path = str(screenshot)
            else:
                artifact_file = tmp_path / "secure_area.png"
                artifact_file.write_bytes(b"secure")
                screenshot_path = str(artifact_file)
            return {
                "completion": {"complete": True, "reason": "secure_area_detected"},
                "error": None,
                "secure_area": {
                    "detected": True,
                    "timestamp": "now",
                    "url": "https://acme.test/secure",
                    "screenshot": screenshot_path,
                },
            }

        async def shutdown(self) -> None:
            self.shutdown_called = True

        def set_mission_context(self, **_: object) -> None:  # noqa: D401 - noop for tests
            return None

    fake_worker = FakeWorker()
    monkeypatch.setattr(MissionExecutor, "_build_worker", lambda self, spec: fake_worker)

    pipeline_calls: list[str] = []

    async def fake_pipeline(
        self,
        *,
        goal_text: str,
        mission_instruction: str,
        dry_run: bool,
        subgoal_dir: Path,
        allow_sensitive: bool,
        worker,
    ) -> dict:  # noqa: ARG001
        pipeline_calls.append(goal_text)
        return {
            "completion": {"complete": True, "reason": "ok"},
            "artifacts": {"base_dir": str(subgoal_dir)},
            "error": None,
        }

    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", fake_pipeline)

    spec = MissionSpec(instruction="Visit https://acme.test/login", execute=False)
    executor = MissionExecutor(
        settings={"planner": {}, "logging": {"artifact_root": str(tmp_path)}}
    )
    result = await executor.run_mission(spec)

    assert result.status == "complete"
    assert fake_worker.payloads[0]["action"][0]["url"] == "https://acme.test/login"
    screenshot_action = fake_worker.payloads[0]["action"][-1]
    assert screenshot_action["action"] == "screenshot"
    assert screenshot_action["name"] == "secure_area.png"
    assert pipeline_calls == []
    assert fake_worker.shutdown_called is True
    assert result.summary["reason"] == "secure_area_detected"
    assert len(result.subgoal_results) == 1
    secure_artifact = result.subgoal_results[0].artifacts["secure_area"]
    screenshot_path = Path(secure_artifact["screenshot"])
    assert screenshot_path.exists()
    assert screenshot_path.name == "secure_area.png"
