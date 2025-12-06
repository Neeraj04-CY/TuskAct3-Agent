from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from eikon_engine.missions.mission_schema import MissionResult, MissionSpec, MissionSubgoalResult


class StubMissionExecutor:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)

    async def run_mission(self, mission_spec: MissionSpec) -> MissionResult:
        now = datetime.now(UTC)
        mission_dir = self.artifacts_dir / f"stub_{mission_spec.id}"
        mission_dir.mkdir(parents=True, exist_ok=True)
        subgoal = MissionSubgoalResult(
            subgoal_id="sg",
            description="stub",
            status="complete",
            attempts=1,
            started_at=now,
            ended_at=now,
        )
        result = MissionResult(
            mission_id=mission_spec.id,
            status="complete",
            start_ts=now,
            end_ts=now,
            subgoal_results=[subgoal],
            summary={"reason": "stub"},
            artifacts_path=str(mission_dir),
        )
        (mission_dir / "mission_result.json").write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
        return result


def build_executor(artifacts_dir: Path) -> StubMissionExecutor:
    return StubMissionExecutor(artifacts_dir)
