import json
from pathlib import Path

import pytest

from eikon_engine.missions.mission_executor import MissionExecutor, StrategyViolationError
from eikon_engine.missions.mission_schema import MissionSpec, MissionSubgoal
from eikon_engine.trace.recorder import ExecutionTraceRecorder


class RecorderStub:
    def __init__(self) -> None:
        self.page_intents: list[dict] = []
        self.skill_usage: list[dict] = []
        self.extractions: list[dict] = []
        self.artifacts: list[dict] = []
        self.skips: list[dict] = []

    def record_page_intent(self, *, intent: str, confidence: float, strategy=None, signals=None, step_id=None) -> None:
        self.page_intents.append({
            "intent": intent,
            "confidence": confidence,
            "strategy": strategy,
            "signals": signals or {},
            "step_id": step_id,
        })

    def record_skill_usage(self, *, name: str, status: str, handle=None, metadata=None, learning_bias=None) -> None:
        self.skill_usage.append({
            "name": name,
            "status": status,
            "handle": handle,
            "metadata": metadata or {},
            "learning_bias": learning_bias,
        })

    def record_extraction(self, *, name: str, status: str, summary, artifact_path=None) -> None:
        self.extractions.append({
            "name": name,
            "status": status,
            "summary": dict(summary),
            "artifact_path": artifact_path,
        })

    def record_artifact(self, name: str, path_value: str) -> None:
        self.artifacts.append({"name": name, "path": path_value})

    def record_subgoal_skip(self, *, subgoal: MissionSubgoal, reason: str, page_intent: str | None = None) -> None:
        self.skips.append({
            "subgoal": subgoal.description,
            "reason": reason,
            "page_intent": page_intent,
        })


class WorkerStub:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.invocations: list[tuple[str, dict]] = []
        self.logger = None
        self.learning_bias = None

    async def run_skill(self, name: str, context: dict) -> dict:
        self.invocations.append((name, dict(context)))
        artifact_path = context.get("artifact_path")
        if artifact_path:
            path = Path(artifact_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = self.response.get("result") or {}
            path.write_text(json.dumps(payload), encoding="utf-8")
        return dict(self.response)

    def set_mission_context(self, **_: dict) -> None:  # pragma: no cover - compatibility shim
        return None

    def set_trace_context(self, **_: dict) -> None:  # pragma: no cover - compatibility shim
        return None

    def clear_trace_context(self) -> None:  # pragma: no cover - compatibility shim
        return None

    def set_learning_bias(self, metadata: dict | None) -> None:  # pragma: no cover - compatibility shim
        self.learning_bias = metadata

    async def shutdown(self) -> None:  # pragma: no cover - compatibility shim
        return None


@pytest.fixture
def executor(tmp_path) -> MissionExecutor:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return MissionExecutor(
        settings={"planner": {}},
        artifacts_root=tmp_path,
        trace_recorder_factory=lambda: ExecutionTraceRecorder(storage_dir=trace_dir),
    )


def test_record_page_intents_emits_trace_entries(executor, tmp_path) -> None:
    recorder = RecorderStub()
    run_ctx = {
        "page_intents": [
            {
                "intent": "listing_page",
                "confidence": 0.82,
                "strategy": "listing_extraction",
                "signals": {"card_repetition": 4},
                "step_id": "nav-1",
            }
        ]
    }

    executor._record_page_intents(run_ctx, recorder, attempt_handle="sg1")

    assert recorder.page_intents[0]["intent"] == "listing_page"
    assert recorder.page_intents[0]["strategy"] == "listing_extraction"


@pytest.mark.asyncio
async def test_maybe_run_listing_extraction_runs_skill_and_records_artifact(executor, tmp_path) -> None:
    mission_spec = MissionSpec(instruction="Find startup listings", execute=True)
    run_ctx = {
        "requested_skills": [{"name": "listing_extraction_skill"}],
        "current_page_intent": {"intent": "listing_page"},
        "current_url": "https://www.ycombinator.com/companies",
    }
    worker = WorkerStub({"status": "success", "result": {"name": "Atlas Robotics"}})
    recorder = RecorderStub()
    subgoal_dir = tmp_path / "subgoal"
    subgoal_dir.mkdir()

    outcome = await executor._maybe_run_listing_extraction(
        mission_spec=mission_spec,
        worker=worker,
        run_ctx=run_ctx,
        subgoal_dir=subgoal_dir,
        trace_recorder=recorder,
        attempt_handle="sg1",
    )

    assert outcome["status"] == "success"
    assert recorder.skill_usage[0]["name"] == "listing_extraction_skill"
    assert recorder.extractions[0]["status"] == "success"
    artifact_path = Path(outcome["artifact"])
    assert artifact_path.exists()
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["name"] == "Atlas Robotics"
    assert worker.invocations and worker.invocations[0][0] == "listing_extraction_skill"


@pytest.mark.asyncio
async def test_maybe_run_listing_extraction_skips_when_not_executing(executor, tmp_path) -> None:
    mission_spec = MissionSpec(instruction="Find startup listings", execute=False)
    run_ctx = {"requested_skills": [{"name": "listing_extraction_skill"}]}
    worker = WorkerStub({"status": "success", "result": {}})
    recorder = RecorderStub()
    subgoal_dir = tmp_path / "subgoal"
    subgoal_dir.mkdir()

    outcome = await executor._maybe_run_listing_extraction(
        mission_spec=mission_spec,
        worker=worker,
        run_ctx=run_ctx,
        subgoal_dir=subgoal_dir,
        trace_recorder=recorder,
        attempt_handle="sg1",
    )

    assert outcome["status"] == "skipped"
    assert recorder.extractions[0]["status"] == "skipped"
    assert worker.invocations == []


def test_subgoal_skip_logged_for_listing_intent(executor) -> None:
    recorder = RecorderStub()
    subgoal = MissionSubgoal(id="sg2", description="01. form: dom_presence_check", planner_metadata={})
    last_run_ctx = {
        "current_page_intent": {"intent": "listing_page"},
    }

    result = executor._maybe_skip_subgoal_for_intent(subgoal=subgoal, last_run_ctx=last_run_ctx, trace_recorder=recorder)

    assert result is not None
    assert result.status == "skipped"
    assert result.completion["reason"] == "page_intent_known"
    assert result.completion["page_intent"] == "LISTING_PAGE"
    payload = result.completion.get("payload")
    assert payload["page_intent"] == "LISTING_PAGE"
    assert recorder.skips[0]["reason"] == "page_intent_known"


def test_navigation_subgoal_skipped_after_listing_intent(executor) -> None:
    recorder = RecorderStub()
    subgoal = MissionSubgoal(id="sg_nav", description="02. navigation: navigate", planner_metadata={})
    last_run_ctx = {"current_page_intent": {"intent": "listing_page"}}

    result = executor._maybe_skip_subgoal_for_intent(subgoal=subgoal, last_run_ctx=last_run_ctx, trace_recorder=recorder)

    assert result is not None
    assert result.status == "skipped"
    assert result.completion["reason"] == "page_intent_known"
    assert recorder.skips[-1]["reason"] == "page_intent_known"


def test_extract_latest_dom_snapshot_prefers_recent_entry(executor) -> None:
    payload = {
        "steps": [
            {"result": {"dom_snapshot": "<html>older</html>"}},
            {"result": {"dom_snapshot": "<html>newer</html>"}},
        ]
    }

    snapshot = executor._extract_latest_dom_snapshot(payload)

    assert snapshot == "<html>newer</html>"


def test_strategy_violation_detected_when_dom_presence_runs_on_listing_page(executor) -> None:
    subgoal = MissionSubgoal(id="sg2", description="01. form: dom_presence_check", planner_metadata={})
    run_ctx = {
        "current_page_intent": {"intent": "listing_page"},
    }

    violation = executor._detect_strategy_violation(subgoal, run_ctx)

    assert violation is not None
    assert violation["page_intent"] == "listing_page"
    assert violation["reason"] == "intent_blocks_form_action"


@pytest.mark.asyncio
async def test_execute_subgoal_raises_strategy_violation(monkeypatch, executor, tmp_path) -> None:
    mission_spec = MissionSpec(instruction="Check listing", execute=True)
    subgoal = MissionSubgoal(id="sg_form", description="01. form: dom_presence_check", planner_metadata={})
    subgoal_dir = tmp_path / "sg_form"
    subgoal_dir.mkdir()
    worker = WorkerStub({"status": "success", "result": {}})

    async def fake_pipeline(self, **_kwargs):  # pragma: no cover - deterministic payload
        return {
            "completion": {"complete": False, "reason": "dom_presence_failed"},
            "artifacts": {},
            "run_context": {
                "current_page_intent": {"intent": "listing_page"},
                "page_intent": {"intent": "listing_page"},
                "page_intents": [{"intent": "listing_page"}],
                "requested_skills": [],
            },
        }

    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", fake_pipeline)

    with pytest.raises(StrategyViolationError):
        await executor._execute_subgoal(
            mission_spec,
            subgoal,
            subgoal_dir,
            worker,
            skill_plan=None,
            used_skills=[],
            detected_url=None,
            trace_recorder=None,
        )


def test_dom_snapshot_probe_records_intent_and_requests_listing_skill(executor) -> None:
    dom_snapshot = """
    <html>
        <body>
            <section class='company card'>Startup Alpha</section>
            <section class='company card'>Startup Beta</section>
            <ul>
                <li class='listing item'><a href='/companies/alpha'>Alpha</a></li>
                <li class='listing item'><a href='/companies/beta'>Beta</a></li>
            </ul>
        </body>
    </html>
    """
    run_ctx = {"current_url": "https://www.ycombinator.com/companies"}
    recorder = RecorderStub()

    executor._capture_page_intent_from_dom(run_ctx=run_ctx, dom_snapshot=dom_snapshot, trace_recorder=recorder)

    assert run_ctx["current_page_intent"]["intent"] == "listing_page"
    assert run_ctx["requested_skills"][0]["name"] == "listing_extraction_skill"
    assert recorder.page_intents[-1]["intent"] == "listing_page"


@pytest.mark.asyncio
async def test_run_mission_skips_form_and_runs_listing_skill(monkeypatch, tmp_path) -> None:
    trace_dir = tmp_path / "full_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    executor = MissionExecutor(
        settings={"planner": {}},
        artifacts_root=tmp_path,
        trace_recorder_factory=lambda: ExecutionTraceRecorder(storage_dir=trace_dir),
    )
    worker = WorkerStub({"status": "success", "result": {"name": "Atlas Robotics"}})

    async def fake_pipeline(self, **_kwargs):  # pragma: no cover - deterministic payload
        return {
            "completion": {"complete": True, "reason": "goal_complete"},
            "artifacts": {},
            "run_context": {
                "current_page_intent": {"intent": "listing_page"},
                "page_intent": {"intent": "listing_page"},
                "page_intents": [{"intent": "listing_page", "confidence": 0.9, "strategy": "listing_extraction", "signals": {}}],
                "requested_skills": [{"name": "listing_extraction_skill"}],
                "current_url": "https://www.ycombinator.com/companies",
            },
        }

    def fake_plan(_mission_spec, settings=None):
        return [
            MissionSubgoal(id="sg1", description="00. navigation: navigate", planner_metadata={}),
            MissionSubgoal(id="sg2", description="01. form: dom_presence_check", planner_metadata={}),
        ]

    monkeypatch.setattr(MissionExecutor, "_build_worker", lambda self, spec: worker)
    monkeypatch.setattr(MissionExecutor, "_run_subgoal_pipeline", fake_pipeline)
    monkeypatch.setattr(MissionExecutor, "_persist_mission_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr("eikon_engine.missions.mission_planner.plan_mission", fake_plan)

    mission_spec = MissionSpec(instruction="Open YC companies and extract a startup", execute=True)
    result = await executor.run_mission(mission_spec)

    assert result.status == "complete"
    assert [sg.status for sg in result.subgoal_results] == ["complete", "skipped"]
    assert worker.invocations and worker.invocations[0][0] == "listing_extraction_skill"

    artifact_candidates = list(Path(result.artifacts_path).rglob("listing_extraction.json"))
    assert artifact_candidates, "listing extraction artifact missing"

    trace_path = Path(result.summary["execution_trace"])
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["page_intents"] and trace["page_intents"][0]["intent"] == "listing_page"
    assert trace["skipped_subgoals"] and trace["skipped_subgoals"][0]["reason"] == "page_intent_known"
    assert trace["extractions"] and trace["extractions"][0]["status"] == "success"
