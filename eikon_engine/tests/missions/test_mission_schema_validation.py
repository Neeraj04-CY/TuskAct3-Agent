from __future__ import annotations

from datetime import datetime, timezone

from eikon_engine.missions.mission_schema import MissionResult, MissionSpec, MissionSubgoalResult, mission_id

UTC = timezone.utc


def test_mission_spec_defaults() -> None:
    spec = MissionSpec(instruction="Collect analytics from dashboard")
    assert spec.id.startswith("mission_")
    assert spec.timeout_secs == 900
    assert spec.max_retries == 2
    assert spec.execute is False
    assert spec.autonomy_budget is None
    assert spec.safety_contract is None
    assert spec.ask_on_uncertainty is False


def test_mission_result_serialization() -> None:
    now = datetime.now(UTC)
    subgoal = MissionSubgoalResult(
        subgoal_id="sg1",
        description="Open dashboard",
        status="complete",
        attempts=1,
        started_at=now,
        ended_at=now,
    )
    result = MissionResult(
        mission_id=mission_id(),
        status="complete",
        start_ts=now,
        end_ts=now,
        subgoal_results=[subgoal],
        summary={"reason": "ok"},
        artifacts_path="/tmp/mission",
    )
    payload = result.model_dump(mode="json")
    assert payload["summary"]["reason"] == "ok"
    assert payload["subgoal_results"][0]["status"] == "complete"
    assert payload["termination"] == {}
