"""Mission execution orchestration built on Strategist V2."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import traceback
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import replace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from eikon_engine.config_loader import load_settings
from eikon_engine.core.orchestrator_v2 import OrchestratorV2
from eikon_engine.memory.memory_store import MissionMemory
from eikon_engine.memory.memory_writer import save_mission_memory
from eikon_engine.approval.models import ApprovalRequest, ApprovalState, UTC as APPROVAL_UTC
from eikon_engine.capabilities.inference import (
    build_plan_capability_report,
    report_to_payload,
    requirements_from_payload,
)
from eikon_engine.capabilities.models import CapabilityRequirement
from eikon_engine.capabilities.registry import capabilities_for_skill, CAPABILITY_REGISTRY
from eikon_engine.capabilities.registry import capabilities_for_skill, CAPABILITY_REGISTRY
from eikon_engine.capabilities.enforcement import (
    CapabilityDecision,
    EnforcementContext,
    evaluate_capabilities,
)
from eikon_engine.judgment.evaluator import JudgmentEvaluator
from eikon_engine.missions import mission_planner as mission_planner_module
from eikon_engine.missions.mission_planner import MissionPlanningError, _build_default_search_url
from eikon_engine.missions.mission_schema import (
    MissionResult,
    MissionSpec,
    MissionStatus,
    MissionSubgoal,
    MissionSubgoalResult,
)
from eikon_engine.missions.models import (
    AutonomyBudget,
    BudgetMonitor,
    DEFAULT_AUTONOMY_BUDGET,
    MissionTermination,
    SafetyContract,
)
from eikon_engine.pipelines.browser_pipeline import PlannerV3Adapter
from eikon_engine.page_intent import PageIntent, classify_page_intent
from eikon_engine.strategist.agent_memory import AgentMemory
from eikon_engine.strategist.strategist_v2 import StrategistV2
from eikon_engine.trace.recorder import ExecutionTraceRecorder
from eikon_engine.trace.models import CapabilityEnforcementDecision
from eikon_engine.trace.summary import write_trace_summary
from eikon_engine.trace.decision_report import write_decision_report
from eikon_engine.learning.decision_explainer import (
    LearningDecisionExplanation,
    build_learning_decision_explanation,
    write_learning_decision_explanation,
)
from eikon_engine.learning.recorder import LearningRecorder
from eikon_engine.learning.index import LearningBias, LearningIndexCache
from eikon_engine.learning.signals import SkillSignal, load_skill_signals
from eikon_engine.learning.diff import emit_learning_artifacts
from eikon_engine.learning.impact_score import LearningImpactScore
from eikon_engine.learning.override_engine import LearningOverrideEngine, PlannerConflict
from eikon_engine.runtime.resume_checkpoint import ResumeCheckpoint
from eikon_engine.runtime.escalation_state import EscalationState
from eikon_engine.missions.source_ranker import rank_source, bucket_source
from eikon_engine.missions.entity_validator import (
    clean_description,
    dedupe_mentions,
    validate_founders,
    validate_website,
)
from eikon_engine.missions.summary_builder import build_executive_summary

UTC = timezone.utc
CONFIRMATION_ACTIONS = {"submit_form", "download_file", "execute_script", "delete_resource"}
from eikon_engine.workers.browser_worker import BrowserWorker
from .artifacts import MissionArtifactLogger

ESCALATION_STEP_BONUS = 10
ESCALATION_RISK_BONUS = 0.6
ESCALATION_TIME_BONUS_SEC = 60.0
ESCALATION_PAGE_DEPTH_BONUS = 2
ESCALATION_TAB_BONUS = 2

LOW_SIGNAL_BASE_DOMAINS = {
    "tracxn.com",
    "zoominfo.com",
    "pitchbook.com",
}

_FOUNDER_TITLE_PREFIXES = (
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sir",
    "madam",
    "ceo",
    "cto",
    "cfo",
    "cofounder",
    "co-founder",
    "founder",
)

FORM_SUBGOAL_KEYWORDS = ("form", "dom_presence_check", "login")
NAV_SUBGOAL_KEYWORDS = ("navigation", "navigate")

SleepFn = Callable[[float], Awaitable[None]]
TraceRecorderFactory = Callable[[], ExecutionTraceRecorder]

plan_mission = mission_planner_module.plan_mission


class StrategyViolationError(RuntimeError):
    """Raised when a subgoal runs despite a conflicting page intent."""

    def __init__(self, *, subgoal: MissionSubgoal, page_intent: str) -> None:
        super().__init__("strategy_violation")
        self.subgoal = subgoal
        self.page_intent = page_intent
        self.details = {
            "subgoal": subgoal.description,
            "subgoal_id": subgoal.id,
            "page_intent": page_intent,
            "reason": "intent_blocks_form_action",
        }
        self.attempts: int = 1
        self.run_ctx: Dict[str, Any] | None = None

SAFE_FORM_INTENTS = {PageIntent.LOGIN_FORM, PageIntent.UNKNOWN}


class MissionExecutor:
    """High-level mission orchestrator built on the browser pipeline."""

    def __init__(
        self,
        *,
        settings: Optional[Dict[str, Any]] = None,
        artifacts_root: Path | str | None = None,
        memory_manager: Optional[AgentMemory] = None,
        logger: Optional[logging.Logger] = None,
        sleep_fn: Optional[SleepFn] = None,
        debug_browser: bool = False,
        trace_recorder_factory: TraceRecorderFactory | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.demo_mode = bool(self.settings.get("demo", False))
        self.artifacts_root = Path(artifacts_root or Path("artifacts"))
        self.memory_manager = memory_manager
        self.logger = logger or logging.getLogger(__name__)
        self._sleep = sleep_fn or asyncio.sleep
        self.debug_browser = debug_browser
        self._trace_recorder_factory = trace_recorder_factory
        learning_cfg = self.settings.get("learning", {}) or {}
        logs_dir = Path(learning_cfg.get("logs_dir", "learning_logs"))
        min_confidence = float(learning_cfg.get("min_confidence", 0.5) or 0.5)
        self.learning_recorder = LearningRecorder(output_dir=logs_dir)
        self._learning_logs_root = logs_dir
        self._learning_bias_enabled = bool(learning_cfg.get("enable_bias", True))
        self._learning_index_cache = LearningIndexCache(root=logs_dir, min_confidence=min_confidence)
        self._learning_threshold = float(learning_cfg.get("override_threshold", 0.0))
        self._learning_hard_floor = float(learning_cfg.get("hard_floor", -0.6))
        self._impact_scorer = LearningImpactScore.from_logs(logs_dir, min_confidence=min_confidence)
        self._capability_enforcement_cfg = self._load_capability_enforcement_settings(self.settings.get("capability_enforcement", {}))
        self._capability_decisions: List[CapabilityDecision] = []
        self._approval_cfg = self._load_approval_settings(self.settings.get("approval", {}))
        self._approval_request_paths: List[str] = []
        self._judgment_evaluator = JudgmentEvaluator()

    async def run_mission(self, mission_spec: MissionSpec, *, resume_from: str | Path | None = None) -> MissionResult:
        """Execute all mission subgoals sequentially (fresh or resumed)."""
        if resume_from is None and (mission_spec.execute or self.debug_browser):
            return await self._run_goal_driven_autonomous_loop(mission_spec)

        resume_checkpoint: ResumeCheckpoint | None = None
        resume_mode = resume_from is not None
        resume_checkpoint_path: Path | None = None
        resumed_at = datetime.now(UTC) if resume_mode else None
        start_ts = datetime.now(UTC)
        if resume_mode:
            resume_checkpoint = self._load_resume_checkpoint(resume_from)
            resume_checkpoint_path = self._checkpoint_path_from_resume(resume_from)
            mission_spec = mission_spec.model_copy(
                update={
                    "id": resume_checkpoint.mission_id,
                    "instruction": self._load_mission_text_from_trace(resume_checkpoint.trace_path)
                    or mission_spec.instruction,
                }
            )
            start_ts = resumed_at or start_ts
            mission_dir = Path(resume_checkpoint.artifacts_path or resume_checkpoint_path.parent)
        else:
            mission_dir = self._build_mission_dir(mission_spec, start_ts)

        escalation_state = EscalationState()
        if resume_checkpoint and resume_checkpoint.escalation_state:
            escalation_state = EscalationState.from_dict(resume_checkpoint.escalation_state)

        trace_recorder = self._build_trace_recorder()
        trace_recorder.start(mission_spec=mission_spec, mission_dir=mission_dir, started_at=start_ts)
        if resume_mode:
            trace_recorder.record_lifecycle_event(event="resume_loaded", data={"checkpoint": str(resume_checkpoint_path)})
        self._capability_decisions = []
        self._approval_request_paths = []
        halt_event_recorded = False
        subgoal_results: List[MissionSubgoalResult] = []
        if resume_mode:
            subgoal_results.extend(self._load_prior_subgoal_results(mission_dir))
        completed_ids = {result.subgoal_id for result in subgoal_results}
        if resume_checkpoint:
            completed_ids.update(resume_checkpoint.completed_subgoals)
        planned_subgoal_ids: List[str] = []
        pending_snapshot: List[str] = []
        resume_checkpoint_written: Path | None = None
        summary: Dict[str, Any] = {}
        halted_subgoal_id: str | None = None
        halted_reason: str | None = None
        status: MissionStatus = "running"
        worker: BrowserWorker | None = None
        detected_url: str | None = None
        used_skills: List[str] = list(resume_checkpoint.skills_used) if resume_checkpoint else []
        learning_bias: LearningBias | None = None
        learning_review_ctx: Dict[str, Any] | None = None
        learning_explanation_path: Path | None = None
        learning_effect_required = False
        baseline_skill_plan: List[str] = []
        escalation_artifacts: Dict[str, str] = {}
        last_run_context: Dict[str, Any] | None = None
        if resume_checkpoint and resume_checkpoint.page_intent:
            last_run_context = {"page_intent": {"intent": resume_checkpoint.page_intent}}
        if resume_checkpoint and resume_checkpoint.page_url:
            last_run_context = (last_run_context or {}) | {"page_url": resume_checkpoint.page_url}
        autonomy_budget = self._resolve_autonomy_budget(mission_spec)
        base_budget = autonomy_budget
        if escalation_state and escalation_state.used and escalation_state.expanded_budget:
            autonomy_budget = AutonomyBudget(
                max_steps=int(escalation_state.expanded_budget.get("max_steps", base_budget.max_steps)),
                max_retries=int(escalation_state.expanded_budget.get("max_retries", base_budget.max_retries)),
                max_duration_sec=float(escalation_state.expanded_budget.get("max_duration_sec", base_budget.max_duration_sec)),
                max_risk_score=float(escalation_state.expanded_budget.get("max_risk_score", base_budget.max_risk_score)),
            )
        budget_monitor = BudgetMonitor(autonomy_budget)
        safety_contract = self._resolve_safety_contract(mission_spec)
        termination_payload: Dict[str, Any] = {}
        mission_result: MissionResult | None = None
        mission_exception: Exception | None = None
        trace_path: Path | None = None
        mission_result_path = mission_dir / "mission_result.json"
        try:
            if mission_spec.constraints and mission_spec.constraints.get("demo_force_actions"):
                await self._run_demo_force_actions(
                    mission_spec=mission_spec,
                    mission_dir=mission_dir,
                    trace_recorder=trace_recorder,
                )
            # cache mission constraints for downstream planner context
            self._mission_constraints = mission_spec.constraints
            capability_report: Dict[str, Any] = self._empty_capability_report()
            capability_report_path: Path | None = None
            try:
                subgoals, capability_report = self._plan_subgoals(mission_spec)
                if resume_mode and resume_checkpoint:
                    subgoals = self._select_subgoals_for_resume(subgoals, resume_checkpoint)
                planned_subgoal_ids = [subgoal.id for subgoal in subgoals]
            except MissionPlanningError as exc:
                status = "failed"
                summary = {"reason": "planner_error", "detail": str(exc)}
                summary["termination"] = {}
                summary["autonomy_budget"] = budget_monitor.snapshot()
                summary["cost_estimate"] = self._estimate_cost(summary["autonomy_budget"])
                summary["reason_summary"] = self._build_reason_summary(status=status, summary=summary, termination={})
                trace_recorder.record_failure(failure_type="planner_error", message=str(exc), retryable=False)
                end_ts = datetime.now(UTC)
                trace_recorder.record_artifact("mission_result", str(mission_dir / "mission_result.json"))
                self._persist_capability_enforcement(
                    mission_dir=mission_dir,
                    mission_id=mission_spec.id,
                    trace_recorder=trace_recorder,
                    summary=summary,
                )
                self._attach_approval_requests(summary)
                self._attach_capability_report(summary, capability_report, capability_report_path)
                trace_path = self._finalize_trace(trace_recorder=trace_recorder, status=status, summary=summary, end_ts=end_ts)
                mission_result = MissionResult(
                    mission_id=mission_spec.id,
                    status=status,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    subgoal_results=subgoal_results,
                    summary=summary,
                    artifacts_path=str(mission_dir),
                    termination={},
                )
                self._write_result_file(mission_dir, mission_result)
                return mission_result

            capability_report_path = self._write_capability_report(
                mission_dir=mission_dir,
                capability_report=capability_report,
            )
            if capability_report_path:
                trace_recorder.record_artifact("capability_report", str(capability_report_path))
            trace_recorder.record_capability_report(capability_report)

            if resume_mode and resume_checkpoint_path:
                summary["resumed_from_checkpoint"] = str(resume_checkpoint_path)
                summary["resumed_at"] = (resumed_at or start_ts).isoformat()
                summary["resume_pending_subgoals"] = list(resume_checkpoint.pending_subgoals) if resume_checkpoint else []

            detected_url = self._detect_primary_url(subgoals)
            memory_hints = StrategistV2.memory_skill_hints(mission_spec.instruction, url=detected_url)
            learning_bias = self._resolve_learning_bias(mission_spec)
            baseline_skill_plan = list(memory_hints or [])
            skill_plan = self._merge_skill_plan(memory_hints, learning_bias)

            subgoals, override_decision, learning_review_ctx = self._learning_review(
                mission_spec=mission_spec,
                subgoals=subgoals,
                learning_bias=learning_bias,
                trace_recorder=trace_recorder,
            )
            if override_decision and override_decision.get("status") == "refused_by_learning":
                learning_effect_required = True
                summary = override_decision.get("summary", {})
                explanation = self._build_learning_explanation(
                    mission_id=mission_spec.id,
                    decision_type="refusal",
                    final_resolution="refused",
                    learning_context=learning_review_ctx,
                    learning_bias=learning_bias,
                    summary_text=self._render_learning_summary(
                        decision_type="refusal",
                        learning_context=learning_review_ctx,
                        learning_bias=learning_bias,
                    ),
                )
                learning_explanation_path = self._emit_learning_explanation(
                    mission_dir=mission_dir,
                    trace_recorder=trace_recorder,
                    explanation=explanation,
                )
                status = "refused_by_learning"
                if learning_effect_required and not (learning_explanation_path and learning_explanation_path.exists()):
                    status = "failed"
                    summary = {
                        "reason": "learning_artifact_missing",
                        "detail": {"expected": str(mission_dir / "learning_decision_explanation.json")},
                    }
                    trace_recorder.record_failure(
                        failure_type="learning_artifact_missing",
                        message="learning_decision_explanation.json was not written",
                        retryable=False,
                    )
                end_ts = datetime.now(UTC)
                self._persist_capability_enforcement(
                    mission_dir=mission_dir,
                    mission_id=mission_spec.id,
                    trace_recorder=trace_recorder,
                    summary=summary,
                )
                self._attach_capability_report(summary, capability_report, capability_report_path)
                trace_path = self._finalize_trace(
                    trace_recorder=trace_recorder,
                    status=status,
                    summary=summary,
                    end_ts=end_ts,
                )
                mission_result = MissionResult(
                    mission_id=mission_spec.id,
                    status=status,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    subgoal_results=subgoal_results,
                    summary=summary,
                    artifacts_path=str(mission_dir),
                    termination={},
                )
                self._write_result_file(mission_dir, mission_result)
                return mission_result

            worker = self._build_worker(mission_spec)
            self._bind_worker_learning_bias(worker, learning_bias)
            deadline = start_ts + timedelta(seconds=mission_spec.timeout_secs)
            for index, subgoal in enumerate(subgoals, start=1):
                if datetime.now(UTC) > deadline:
                    status = "failed"
                    summary = {"reason": "timeout", "failed_subgoal": subgoal.id}
                    trace_recorder.record_failure(
                        failure_type="mission_timeout",
                        message="mission exceeded deadline",
                        retryable=False,
                    )
                    break
                if resume_checkpoint:
                    if resume_checkpoint.completed_subgoals and subgoal.id in resume_checkpoint.completed_subgoals:
                        continue
                    if resume_checkpoint.pending_subgoals and subgoal.id not in resume_checkpoint.pending_subgoals:
                        continue
                capability_requirements = self._capability_requirements_for_subgoal(subgoal)
                capability_enforcements = self._evaluate_capability_enforcements(
                    requirements=capability_requirements,
                    subgoal_id=subgoal.id,
                )
                trace_capability_enforcements = self._to_trace_capability_enforcements(capability_enforcements)
                if capability_enforcements:
                    self._capability_decisions.extend(capability_enforcements)
                judgment_decision = self._evaluate_judgment(
                    mission_spec=mission_spec,
                    subgoal=subgoal,
                    capability_requirements=capability_requirements,
                    safety_contract=safety_contract,
                    learning_bias=learning_bias,
                    last_page_intent=(last_run_context or {}).get("page_intent") if last_run_context else None,
                )
                if judgment_decision.decision == "halt":
                    status = "halted"
                    summary = {
                        "reason": "judgment_refusal",
                        "detail": judgment_decision.explanation,
                    }
                    halted_subgoal_id = subgoal.id
                    halted_reason = summary["reason"]
                    termination_payload = self._build_termination_payload(
                        termination_type=MissionTermination.HALTED,
                        reason="judgment_refusal",
                        detail={"risk_factors": judgment_decision.risk_factors},
                        budget_snapshot=budget_monitor.snapshot(),
                    )
                    judgment_paths = self._persist_judgment_artifacts(
                        mission_dir=mission_dir,
                        decision=judgment_decision,
                        subgoal=subgoal,
                        trace_recorder=trace_recorder,
                    )
                    summary.update(judgment_paths)
                    self._attach_approval_requests(summary)
                    self._attach_capability_report(summary, capability_report, capability_report_path)
                    if trace_recorder:
                        trace_recorder.record_lifecycle_event(
                            event="mission_halted",
                            data={"subgoal_id": subgoal.id, "reason": "judgment_refusal"},
                        )
                        halt_event_recorded = True
                        trace_recorder.record_warning("Agent halted execution due to autonomous judgment.")
                    break
                if judgment_decision.decision == "request_approval":
                    approval_payload = self._emit_judgment_approval_request(
                        mission_dir=mission_dir,
                        mission_spec=mission_spec,
                        subgoal=subgoal,
                        decision=judgment_decision,
                        capability_requirements=capability_requirements,
                        trace_recorder=trace_recorder,
                    )
                    status = "ask_human"
                    summary = {
                        "reason": "judgment_request_approval",
                        "detail": judgment_decision.explanation,
                        "approval_request_path": approval_payload.get("approval_request_path"),
                        "approval_requests": approval_payload.get("approval_requests"),
                    }
                    halted_subgoal_id = subgoal.id
                    halted_reason = summary["reason"]
                    termination_payload = self._build_termination_payload(
                        termination_type=MissionTermination.ASK_HUMAN,
                        reason="judgment_request_approval",
                        detail={"risk_factors": judgment_decision.risk_factors},
                        budget_snapshot=budget_monitor.snapshot(),
                    )
                    self._attach_approval_requests(summary)
                    self._attach_capability_report(summary, capability_report, capability_report_path)
                    if trace_recorder:
                        trace_recorder.record_lifecycle_event(
                            event="mission_halted",
                            data={"subgoal_id": subgoal.id, "reason": "judgment_request_approval"},
                        )
                        halt_event_recorded = True
                        trace_recorder.record_warning("Agent paused for human approval due to judgment evaluation.")
                    break
                approval_state, approval_reason, approval_path = await self._maybe_request_approval(
                    mission_dir=mission_dir,
                    mission_spec=mission_spec,
                    subgoal=subgoal,
                    capability_requirements=capability_requirements,
                    capability_enforcements=capability_enforcements,
                    trace_recorder=trace_recorder,
                    learning_bias=learning_bias,
                )
                if approval_path:
                    trace_recorder.record_artifact("approval_request", str(approval_path))
                    self._approval_request_paths.append(str(approval_path))
                if approval_state == "rejected":
                    status = "halted"
                    termination_payload = self._build_termination_payload(
                        termination_type=MissionTermination.HALTED,
                        reason="approval_rejected",
                        detail={"reason": approval_reason or "approval_rejected"},
                        budget_snapshot=budget_monitor.snapshot(),
                    )
                    summary = {
                        "reason": "approval_rejected",
                        "detail": {"reason": approval_reason},
                        "approval_request_path": str(approval_path) if approval_path else None,
                    }
                    halted_subgoal_id = subgoal.id
                    halted_reason = summary["reason"]
                    break
                if approval_state == "expired":
                    status = "ask_human"
                    termination_payload = self._build_termination_payload(
                        termination_type=MissionTermination.ASK_HUMAN,
                        reason="approval_expired",
                        detail={"reason": approval_reason or "approval_expired"},
                        budget_snapshot=budget_monitor.snapshot(),
                    )
                    summary = {
                        "reason": "approval_expired",
                        "detail": {"reason": approval_reason},
                        "approval_request_path": str(approval_path) if approval_path else None,
                    }
                    halted_subgoal_id = subgoal.id
                    halted_reason = summary["reason"]
                    break
                subgoal_dir = mission_dir / f"subgoal_{index:02d}"
                subgoal_dir.mkdir(parents=True, exist_ok=True)
                trace_recorder.record_artifact(f"subgoal_{index:02d}", str(subgoal_dir))
                skip_result = self._maybe_skip_subgoal_for_intent(
                    subgoal=subgoal,
                    last_run_ctx=last_run_context,
                    trace_recorder=trace_recorder,
                )
                if skip_result:
                    if trace_recorder and trace_capability_enforcements:
                        trace_recorder.record_capability_enforcements(handle=None, decisions=trace_capability_enforcements)
                    subgoal_results.append(skip_result)
                    completed_ids.add(skip_result.subgoal_id)
                    continue
                try:
                    result, run_ctx, termination_signal = await self._execute_subgoal(
                        mission_spec,
                        subgoal,
                        subgoal_dir,
                        worker,
                        skill_plan=skill_plan,
                        used_skills=used_skills,
                        detected_url=detected_url,
                        trace_recorder=trace_recorder,
                        budget_monitor=budget_monitor,
                        safety_contract=safety_contract,
                        learning_bias=learning_bias,
                        capability_enforcements=trace_capability_enforcements,
                    )
                except StrategyViolationError as violation_exc:
                    status = "failed"
                    detail = dict(violation_exc.details)
                    summary = {"reason": "strategy_violation", "detail": detail}
                    trace_recorder.record_failure(
                        failure_type="strategy_violation",
                        message=json.dumps(detail),
                        retryable=False,
                    )
                    violation_result = self._build_strategy_violation_result(
                        subgoal=subgoal,
                        attempts=getattr(violation_exc, "attempts", 1),
                        page_intent=violation_exc.page_intent,
                    )
                    subgoal_results.append(violation_result)
                    completed_ids.add(violation_result.subgoal_id)
                    break
                subgoal_results.append(result)
                completed_ids.add(result.subgoal_id)
                if termination_signal:
                    status = termination_signal.get("status", "halted")  # type: ignore[assignment]
                    termination_type = MissionTermination.HALTED if status == "halted" else MissionTermination.ASK_HUMAN
                    termination_payload = self._build_termination_payload(
                        termination_type=termination_type,
                        reason=termination_signal.get("reason", "safety_contract"),
                        detail=termination_signal.get("detail"),
                        budget_snapshot=budget_monitor.snapshot(),
                    )
                    halted_subgoal_id = subgoal.id
                    halted_reason = termination_signal.get("reason", "safety_contract")
                    summary = {
                        "reason": termination_signal.get("reason", "safety_contract"),
                        "detail": termination_signal.get("detail", {}),
                    }
                    summary["termination"] = termination_payload
                    summary["autonomy_budget"] = budget_monitor.snapshot()
                    summary["cost_estimate"] = self._estimate_cost(summary["autonomy_budget"])
                    summary["reason_summary"] = self._build_reason_summary(
                        status=status,
                        summary=summary,
                        termination=termination_payload,
                    )
                    if trace_recorder:
                        trace_recorder.record_failure(
                            failure_type="safety_contract_violation",
                            message=json.dumps(termination_payload.get("detail", {})),
                            retryable=False,
                        )
                        trace_recorder.record_lifecycle_event(
                            event="mission_halted",
                            data={
                                "subgoal_id": subgoal.id,
                                "reason": termination_signal.get("reason", "safety_contract"),
                            },
                        )
                        halt_event_recorded = True
                    break
                if run_ctx:
                    last_run_context = run_ctx
                violation = self._detect_strategy_violation(subgoal, run_ctx)
                if violation:
                    status = "failed"
                    summary = {"reason": "strategy_violation", "detail": violation}
                    trace_recorder.record_failure(
                        failure_type="strategy_violation",
                        message=json.dumps(violation),
                        retryable=False,
                    )
                    break
                secure_artifact = result.artifacts.get("secure_area") if result.artifacts else None
                if secure_artifact and secure_artifact.get("detected"):
                    status = "complete"
                    summary = {
                        "reason": "secure_area_detected",
                        "subgoals_completed": len(subgoal_results),
                        "secure_area": secure_artifact,
                    }
                    break  # secure area success termination condition
                if result.status != "complete":
                    status = "failed"
                    summary = {
                        "reason": "subgoal_failed",
                        "failed_subgoal": subgoal.id,
                        "error": result.error,
                    }
                    trace_recorder.record_failure(
                        failure_type="mission_subgoal_failed",
                        message=result.error or "subgoal_failed",
                        retryable=False,
                    )
                    break
                exceeded, limit_reason, limit_detail = budget_monitor.limits_exceeded()
                if exceeded:
                    if (
                        limit_reason == "risk_budget_exceeded"
                        and escalation_state.allowed
                        and not escalation_state.used
                    ):
                        expanded_budget = self._compute_escalation_budget(base_budget)
                        window_limits = {
                            "time_limit_sec": ESCALATION_TIME_BONUS_SEC,
                            "step_bonus": ESCALATION_STEP_BONUS,
                            "risk_bonus": ESCALATION_RISK_BONUS,
                            "tab_bonus": ESCALATION_TAB_BONUS,
                            "page_depth_bonus": ESCALATION_PAGE_DEPTH_BONUS,
                        }
                        escalation_artifacts = self._enter_escalation(
                            mission_dir=mission_dir,
                            trace_recorder=trace_recorder,
                            escalation_state=escalation_state,
                            budget_monitor=budget_monitor,
                            base_budget=base_budget,
                            expanded_budget=expanded_budget,
                            window_limits=window_limits,
                            limit_detail=limit_detail,
                        )
                        continue
                    status = "halted"
                    termination_payload = self._build_termination_payload(
                        termination_type=MissionTermination.HALTED,
                        reason="autonomy_budget",
                        detail={"code": limit_reason, **limit_detail},
                        budget_snapshot=budget_monitor.snapshot(),
                    )
                    summary = {
                        "reason": "autonomy_budget_exceeded",
                        "detail": termination_payload.get("detail"),
                    }
                    if escalation_state.used:
                        summary["escalation_used"] = True
                    halted_subgoal_id = subgoal.id
                    halted_reason = summary["reason"]
                    summary["termination"] = termination_payload
                    summary["autonomy_budget"] = termination_payload.get("budget_snapshot", {})
                    summary["cost_estimate"] = self._estimate_cost(summary["autonomy_budget"])
                    summary["reason_summary"] = self._build_reason_summary(
                        status=status,
                        summary=summary,
                        termination=termination_payload,
                    )
                    budget_monitor.record_failure()
                    trace_recorder.record_failure(
                        failure_type="autonomy_budget_exceeded",
                        message=json.dumps(limit_detail),
                        retryable=False,
                    )
                    break
                if mission_spec.ask_on_uncertainty and self._should_escalate_on_uncertainty(budget_monitor):
                    status = "ask_human"
                    termination_payload = self._build_termination_payload(
                        termination_type=MissionTermination.ASK_HUMAN,
                        reason="low_confidence",
                        detail={
                            "average_confidence": budget_monitor.usage.average_confidence(),
                            "risk_score": budget_monitor.usage.risk_score,
                        },
                        budget_snapshot=budget_monitor.snapshot(),
                    )
                    summary = {
                        "reason": "ask_on_uncertainty",
                        "detail": termination_payload.get("detail"),
                    }
                    halted_subgoal_id = subgoal.id
                    halted_reason = summary["reason"]
                    summary["termination"] = termination_payload
                    summary["autonomy_budget"] = termination_payload.get("budget_snapshot", {})
                    summary["cost_estimate"] = self._estimate_cost(summary["autonomy_budget"])
                    summary["reason_summary"] = self._build_reason_summary(
                        status=status,
                        summary=summary,
                        termination=termination_payload,
                    )
                    budget_monitor.record_failure()
                    trace_recorder.record_failure(
                        failure_type="ask_on_uncertainty",
                        message=json.dumps(summary["detail"] or {}),
                        retryable=False,
                    )
                    break
            else:
                if status == "running":
                    status = "complete"
                    summary = {
                        "reason": "mission_complete",
                        "subgoals_completed": len(subgoal_results),
                    }
            if status in {"halted", "ask_human"}:
                pending_snapshot = [sid for sid in planned_subgoal_ids if sid not in completed_ids]
                summary["pending_subgoals"] = pending_snapshot
                if halted_subgoal_id:
                    summary["halted_subgoal_id"] = halted_subgoal_id
                if trace_recorder and not halt_event_recorded:
                    trace_recorder.record_lifecycle_event(
                        event="mission_halted",
                        data={
                            "subgoal_id": halted_subgoal_id or (pending_snapshot[0] if pending_snapshot else None),
                            "reason": summary.get("reason"),
                        },
                    )
                    halt_event_recorded = True
            self._check_escalation_window(
                mission_dir=mission_dir,
                trace_recorder=trace_recorder,
                escalation_state=escalation_state,
                budget_monitor=budget_monitor,
                base_budget=base_budget,
                artifacts=escalation_artifacts,
            )
            end_ts = datetime.now(UTC)
            result_file = mission_dir / "mission_result.json"
            trace_recorder.record_artifact("mission_result", str(result_file))
            budget_snapshot = budget_monitor.snapshot()
            summary.setdefault("autonomy_budget", budget_snapshot)
            summary.setdefault("cost_estimate", self._estimate_cost(summary["autonomy_budget"]))
            if termination_payload:
                summary["termination"] = termination_payload
            summary.setdefault("termination", termination_payload)
            if learning_review_ctx or learning_bias:
                learning_effect_required = self._learning_altered_execution(
                    learning_bias=learning_bias,
                    learning_context=learning_review_ctx,
                    baseline_skill_plan=baseline_skill_plan,
                    merged_skill_plan=skill_plan,
                )
            if learning_effect_required and learning_explanation_path is None:
                decision_type = "override" if self._learning_override_applied(learning_review_ctx) else "bias_applied"
                final_resolution = "override_applied" if decision_type == "override" else "bias_only"
                explanation = self._build_learning_explanation(
                    mission_id=mission_spec.id,
                    decision_type=decision_type,
                    final_resolution=final_resolution,
                    learning_context=learning_review_ctx,
                    learning_bias=learning_bias,
                    summary_text=self._render_learning_summary(
                        decision_type=decision_type,
                        learning_context=learning_review_ctx,
                        learning_bias=learning_bias,
                    ),
                )
                learning_explanation_path = self._emit_learning_explanation(
                    mission_dir=mission_dir,
                    trace_recorder=trace_recorder,
                    explanation=explanation,
                )
            if learning_effect_required and not (learning_explanation_path and learning_explanation_path.exists()):
                status = "failed"
                summary = {
                    "reason": "learning_artifact_missing",
                    "detail": {"expected": str(mission_dir / "learning_decision_explanation.json")},
                }
                termination_payload = {}
                trace_recorder.record_failure(
                    failure_type="learning_artifact_missing",
                    message="learning_decision_explanation.json was not written",
                    retryable=False,
                )
            self._persist_capability_enforcement(
                mission_dir=mission_dir,
                mission_id=mission_spec.id,
                trace_recorder=trace_recorder,
                summary=summary,
            )
            self._attach_approval_requests(summary)
            self._attach_capability_report(summary, capability_report, capability_report_path)
            summary.setdefault(
                "reason_summary",
                self._build_reason_summary(status=status, summary=summary, termination=termination_payload),
            )
            summary["escalation_state"] = escalation_state.to_dict()
            if escalation_state.used and not escalation_state.ended_at:
                escalation_state.mark_closed(summary.get("reason") or status)
                if "escalation_window" in escalation_artifacts:
                    window_payload = json.loads(Path(escalation_artifacts["escalation_window"]).read_text(encoding="utf-8"))
                else:
                    window_payload = {}
                window_payload.update(
                    {
                        "ended_at": escalation_state.ended_at,
                        "closed_reason": summary.get("reason") or status,
                    }
                )
                escalation_artifacts["escalation_window"] = self._write_json(
                    mission_dir / "escalation_window.json", window_payload
                )
                if trace_recorder:
                    trace_recorder.record_lifecycle_event(
                        event="escalation_closed",
                        data={"ended_at": escalation_state.ended_at, "reason": summary.get("reason") or status},
                    )
            if escalation_artifacts:
                summary["escalation_artifacts"] = escalation_artifacts
            if escalation_state.used:
                required_keys = {
                    "escalation_request",
                    "escalation_decision",
                    "escalation_window",
                    "escalation_summary",
                }
                if not required_keys.issubset(set(escalation_artifacts.keys())):
                    status = "failed"
                    summary["reason"] = "escalation_artifacts_missing"
                    summary["missing_escalation_artifacts"] = sorted(required_keys - set(escalation_artifacts.keys()))
            if resume_mode and status == "complete" and trace_recorder:
                trace_recorder.record_lifecycle_event(event="resume_completed", data={"resumed": True})
            if status in {"halted", "ask_human"}:
                checkpoint = ResumeCheckpoint(
                    mission_id=mission_spec.id,
                    halted_subgoal_id=halted_subgoal_id or (pending_snapshot[0] if pending_snapshot else "unknown"),
                    halted_reason=halted_reason or summary.get("reason", "unknown"),
                    page_url=(last_run_context or {}).get("page_url"),
                    page_intent=(self._resolve_page_intent(last_run_context) or PageIntent.UNKNOWN).value if last_run_context else None,
                    completed_subgoals=sorted(completed_ids),
                    pending_subgoals=pending_snapshot or [sid for sid in planned_subgoal_ids if sid not in completed_ids],
                    skills_used=list(used_skills),
                    capability_state={
                        "decisions": summary.get("capability_enforcement", []),
                        "report": summary.get("capability_report", capability_report),
                    },
                    learning_bias_snapshot=self._learning_bias_metadata(learning_bias) or {},
                    trace_path=str(self._predict_trace_path(trace_recorder)),
                    timestamp_utc=datetime.now(UTC).isoformat(),
                    escalation_state=escalation_state.to_dict(),
                    mission_instruction=mission_spec.instruction,
                    artifacts_path=str(mission_dir),
                )
                resume_checkpoint_written = self._persist_resume_checkpoint(
                    mission_dir=mission_dir,
                    checkpoint=checkpoint,
                    trace_recorder=trace_recorder,
                )
                summary["resume_checkpoint"] = str(resume_checkpoint_written)
            trace_path = self._finalize_trace(trace_recorder=trace_recorder, status=status, summary=summary, end_ts=end_ts)
            mission_result = MissionResult(
                mission_id=mission_spec.id,
                status=status,
                start_ts=start_ts,
                end_ts=end_ts,
                subgoal_results=subgoal_results,
                summary=summary,
                artifacts_path=str(mission_dir),
                termination=termination_payload,
            )
            self._write_result_file(mission_dir, mission_result)
            self._record_memory(summary)
            self._persist_mission_memory(
                mission_result=mission_result,
                mission_spec=mission_spec,
                detected_url=detected_url,
                used_skills=used_skills,
                artifacts_dir=mission_dir,
            )
            return mission_result
        except Exception as exc:
            mission_exception = exc
            failure_summary = {"reason": "unhandled_exception", "error": str(exc)}
            try:
                trace_recorder.record_failure(
                    failure_type="unhandled_exception",
                    message=str(exc),
                    retryable=False,
                )
                self._persist_capability_enforcement(
                    mission_dir=mission_dir,
                    mission_id=mission_spec.id,
                    trace_recorder=trace_recorder,
                    summary=failure_summary,
                )
                trace_path = self._finalize_trace(
                    trace_recorder=trace_recorder,
                    status="failed",
                    summary=failure_summary,
                    end_ts=datetime.now(UTC),
                )
            except Exception:
                self.logger.exception("Failed to persist execution trace after crash")
            try:
                mission_result = MissionResult(
                    mission_id=mission_spec.id,
                    status="failed",
                    start_ts=start_ts,
                    end_ts=datetime.now(UTC),
                    subgoal_results=subgoal_results,
                    summary=failure_summary,
                    artifacts_path=str(mission_dir),
                    termination={},
                )
                self._write_result_file(mission_dir, mission_result)
            except Exception:
                self.logger.warning("mission result persistence failed after exception", exc_info=True)
            raise
        finally:
            if worker:
                if self.debug_browser and mission_spec.execute:
                    message = "[DEBUG] Browser staying open for manual inspection"
                    self.logger.info(message)
                    print(message)
                    await getattr(worker, "await_manual_close", worker.shutdown)()
                else:
                    await worker.shutdown()
            try:
                if not (summary and summary.get("reason") in {"approval_expired", "approval_rejected"}):
                    self._record_learning(
                        mission_result_path=mission_result_path,
                        mission_instruction=mission_spec.instruction,
                        trace_path=trace_path,
                        mission_exception=mission_exception,
                    )
            except Exception:  # pragma: no cover - defensive logging
                self.logger.warning("learning record write failed", exc_info=True)

    def _build_mission_dir(self, mission_spec: MissionSpec, start_ts: datetime) -> Path:
        slug = _slugify(mission_spec.instruction)[:40]
        mission_dir = self.artifacts_root / f"mission_{start_ts.strftime('%Y%m%d_%H%M%S')}_{slug}"
        mission_dir.mkdir(parents=True, exist_ok=True)
        return mission_dir

    async def _execute_subgoal(
        self,
        mission_spec: MissionSpec,
        subgoal: MissionSubgoal,
        subgoal_dir: Path,
        worker: BrowserWorker,
        *,
        skill_plan: List[str] | None = None,
        used_skills: List[str] | None = None,
        detected_url: str | None = None,
        trace_recorder: ExecutionTraceRecorder | None = None,
        budget_monitor: BudgetMonitor | None = None,
        safety_contract: SafetyContract | None = None,
        learning_bias: LearningBias | None = None,
        capability_enforcements: List[CapabilityEnforcementDecision] | None = None,
    ) -> Tuple[MissionSubgoalResult, Dict[str, Any] | None, Dict[str, Any] | None]:
        attempts = 0
        last_error: str | None = None
        completion_payload: Dict[str, Any] | None = None
        artifacts: Dict[str, Any] = {}
        start_time = datetime.now(UTC)
        bootstrap_actions = subgoal.planner_metadata.get("bootstrap_actions") if worker else None
        default_attempts = max(1, mission_spec.max_retries + 1)
        subgoal_retry = getattr(subgoal, "retry_limit", None)
        computed_attempts = subgoal_retry if isinstance(subgoal_retry, int) and subgoal_retry > 0 else default_attempts
        max_attempts = 1 if self.demo_mode else computed_attempts
        skill_result: Dict[str, Any] | None = None
        skill_invoked = False
        run_ctx_output: Dict[str, Any] | None = None
        attempt_handle: str | None = None
        termination_signal: Dict[str, Any] | None = None
        while attempts < max_attempts:
            attempts += 1
            attempt_handle = (
                trace_recorder.start_subgoal(
                    subgoal=subgoal,
                    attempt_number=attempts,
                    learning_bias=self._learning_bias_metadata(
                        learning_bias,
                        context={"subgoal": subgoal.description, "attempt": attempts},
                    ),
                    learning_score=(subgoal.planner_metadata.get("learning_override", {}) if subgoal.planner_metadata else {}).get("learning_score"),
                    capability_requirements=(subgoal.planner_metadata or {}).get("capability_requirements"),
                )
                if trace_recorder
                else None
            )
            if trace_recorder and capability_enforcements and attempts == 1:
                trace_recorder.record_capability_enforcements(
                    handle=attempt_handle,
                    decisions=capability_enforcements,
                )
            self._bind_worker_trace(worker, trace_recorder, attempt_handle)
            attempt_status: MissionStatus = "failed"
            try:
                if not skill_invoked and self._should_apply_login_skill(subgoal, mission_spec, skill_plan, used_skills):
                    skill_invoked = True
                    try:
                        skill_result = await self._invoke_login_skill(worker=worker, mission_spec=mission_spec, url=detected_url)
                        if used_skills is not None:
                            used_skills.append("login_form_skill")
                        if skill_result.get("result", {}).get("status") == "success":
                            if trace_recorder:
                                trace_recorder.record_skill_usage(
                                    name="login_form_skill",
                                    status="success",
                                    handle=attempt_handle,
                                    metadata={"payload": skill_result},
                                    learning_bias=self._learning_bias_metadata(
                                        learning_bias,
                                        skill_name="login_form_skill",
                                    ),
                                )
                                self._record_capability_usage_for_skill(
                                    trace_recorder=trace_recorder,
                                    handle=attempt_handle,
                                    skill_id="login_form_skill",
                                    success=True,
                                )
                            completion_payload = {"complete": True, "reason": "skill:login_form_skill"}
                            artifacts = {"skill_login_form_skill": skill_result}
                            last_error = None
                            attempt_status = "complete"
                            break
                        status_label = skill_result.get("result", {}).get("status", "unknown")
                        if trace_recorder:
                            trace_recorder.record_skill_usage(
                                name="login_form_skill",
                                status=status_label,
                                handle=attempt_handle,
                                metadata={"payload": skill_result},
                                learning_bias=self._learning_bias_metadata(
                                    learning_bias,
                                    skill_name="login_form_skill",
                                ),
                            )
                    except Exception as exc:  # pragma: no cover - skill execution best effort
                        self.logger.warning("login skill execution failed", exc_info=True)
                        last_error = str(exc)
                        attempt_status = "failed"
                        completion_payload = None
                        artifacts = {"skill_login_form_skill_error": last_error}
                        if trace_recorder:
                            trace_recorder.record_skill_usage(
                                name="login_form_skill",
                                status="error",
                                handle=attempt_handle,
                                metadata={"reason": "exception", "error": last_error},
                                learning_bias=self._learning_bias_metadata(
                                    learning_bias,
                                    skill_name="login_form_skill",
                                ),
                            )
                            trace_recorder.record_failure(
                                failure_type="login_skill_error",
                                message=last_error,
                                handle=attempt_handle,
                                retryable=False,
                            )
                        break
                if bootstrap_actions:
                    payload = await self._execute_bootstrap_actions(
                        goal_text=subgoal.description,
                        actions=bootstrap_actions,
                        subgoal_dir=subgoal_dir,
                        worker=worker,
                    )
                else:
                    payload = await self._run_subgoal_pipeline(
                        goal_text=subgoal.description,
                        mission_instruction=mission_spec.instruction,
                        dry_run=not mission_spec.execute,
                        subgoal_dir=subgoal_dir,
                        allow_sensitive=mission_spec.allow_sensitive,
                        worker=worker,
                        learning_bias=learning_bias,
                    )
                run_ctx = payload.get("run_context") or {}
                dom_snapshot = self._extract_latest_dom_snapshot(payload)
                self._capture_page_intent_from_dom(
                    run_ctx=run_ctx,
                    dom_snapshot=dom_snapshot,
                    trace_recorder=trace_recorder,
                    budget_monitor=budget_monitor,
                )
                run_ctx_output = run_ctx or run_ctx_output
                self._record_page_intents(run_ctx, trace_recorder, attempt_handle)
                self._record_budget_confidence(run_ctx, budget_monitor)
                try:
                    self._enforce_intent_alignment(
                        subgoal=subgoal,
                        run_ctx=run_ctx,
                    )
                except StrategyViolationError as exc:
                    exc.attempts = attempts
                    exc.run_ctx = run_ctx
                    self._clear_worker_trace(worker)
                    if trace_recorder and attempt_handle:
                        trace_recorder.end_subgoal(
                            handle=attempt_handle,
                            status="failed",
                            error="strategy_violation",
                        )
                    attempt_handle = None
                    raise
                listing_outcome = await self._maybe_run_listing_extraction(
                    mission_spec=mission_spec,
                    worker=worker,
                    run_ctx=run_ctx,
                    subgoal_dir=subgoal_dir,
                    trace_recorder=trace_recorder,
                    attempt_handle=attempt_handle,
                    learning_bias=learning_bias,
                )
                if budget_monitor:
                    step_count = self._count_steps(payload)
                    if step_count:
                        budget_monitor.record_steps(step_count)
                if safety_contract:
                    violation = self._check_safety_contract(payload.get("steps"), safety_contract)
                    if violation:
                        termination_signal = violation
                        attempt_status = violation.get("status", "halted")  # type: ignore[assignment]
                        last_error = violation.get("reason") or "safety_contract"
                        completion_payload = {
                            "complete": False,
                            "reason": last_error,
                            "payload": violation,
                        }
                        break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                completion_payload = None
                artifacts = {"traceback": traceback.format_exc()}
            else:
                completion_payload = payload.get("completion")
                artifacts = dict(payload.get("artifacts", {}))
                error = payload.get("error")
                secure_payload = payload.get("secure_area")
                if secure_payload:
                    artifacts["secure_area"] = secure_payload
                last_error = error or completion_payload.get("reason") if completion_payload else error
                if listing_outcome:
                    outcome_status = listing_outcome.get("status")
                    if outcome_status == "success":
                        completion_payload = listing_outcome.get("completion") or completion_payload
                        artifacts.update(listing_outcome.get("artifacts", {}))
                        last_error = None
                        attempt_status = "complete"
                        payload["completion"] = completion_payload
                    elif outcome_status and outcome_status not in {"skipped", "success"}:
                        last_error = listing_outcome.get("error") or last_error
                if completion_payload and completion_payload.get("complete") and not error:
                    last_error = None
                    attempt_status = "complete"
                    break
                if last_error is None:
                    last_error = "unknown_error"
            if mission_spec.execute and not self.demo_mode:
                delay = 2 ** (attempts - 1)
                await self._sleep(delay)
            self._clear_worker_trace(worker)
            if trace_recorder and attempt_handle:
                trace_recorder.end_subgoal(handle=attempt_handle, status=attempt_status, error=last_error)
                if attempt_status != "complete":
                    trace_recorder.record_failure(
                        failure_type="subgoal_attempt_failed",
                        message=last_error or "unknown_error",
                        handle=attempt_handle,
                        retryable=attempts < max_attempts,
                    )
                    if budget_monitor:
                        budget_monitor.record_failure()
                    if budget_monitor and attempts < max_attempts and termination_signal is None:
                        budget_monitor.record_retry()
                attempt_handle = None
        self._clear_worker_trace(worker)
        if trace_recorder and attempt_handle:
            final_status: MissionStatus
            if termination_signal:
                final_status = termination_signal.get("status", "halted")  # type: ignore[assignment]
            else:
                final_status = "complete" if last_error is None else "failed"
            trace_recorder.end_subgoal(
                handle=attempt_handle,
                status=final_status,
                error=last_error,
            )
            if final_status != "complete":
                trace_recorder.record_failure(
                    failure_type="subgoal_attempt_failed",
                    message=last_error or final_status,
                    handle=attempt_handle,
                    retryable=False,
                )
                if budget_monitor:
                    budget_monitor.record_failure()
        end_time = datetime.now(UTC)
        if termination_signal:
            status: MissionStatus = termination_signal.get("status", "halted")  # type: ignore[assignment]
        else:
            status = "complete" if last_error is None else "failed"
        if skill_result:
            artifacts.setdefault("skill_login_form_skill", skill_result)
        return MissionSubgoalResult(
            subgoal_id=subgoal.id,
            description=subgoal.description,
            status=status,
            attempts=attempts,
            started_at=start_time,
            ended_at=end_time,
            completion=completion_payload,
            error=last_error,
            artifacts=artifacts,
        ), run_ctx_output, termination_signal

    async def _run_subgoal_pipeline(
        self,
        *,
        goal_text: str,
        mission_instruction: str,
        dry_run: bool,
        subgoal_dir: Path,
        allow_sensitive: bool,
        worker: BrowserWorker,
        learning_bias: LearningBias | None = None,
    ) -> Dict[str, Any]:
        planner_context = dict(self.settings.get("planner", {}))
        mission_constraints = getattr(self, "_mission_constraints", {}) or {}
        default_url = None
        if isinstance(mission_constraints, dict):
            default_url = mission_constraints.get("default_url")
            known_urls = mission_constraints.get("known_urls")
            if known_urls:
                planner_context.setdefault("known_urls", known_urls)
        if not default_url:
            default_url = _build_default_search_url(mission_instruction)
        if default_url:
            planner_context.setdefault("default_url", default_url)
            setattr(worker, "default_url", default_url)
        planner = PlannerV3Adapter(context=planner_context)
        strategist = StrategistV2(planner=planner)
        strategist.attach_learning_bias(learning_bias)
        run_logger = MissionArtifactLogger(base_dir=subgoal_dir, goal_name=goal_text)
        worker.logger = run_logger
        worker.set_mission_context(
            mission_instruction=mission_instruction,
            subgoal_description=goal_text,
        )
        orchestrator = OrchestratorV2(
            strategist=strategist,
            worker=worker,
            logger=run_logger,
        )
        result = await orchestrator.run_goal(goal_text)
        metadata = result.setdefault("metadata", {})
        metadata["allow_sensitive"] = allow_sensitive
        metadata["dry_run"] = dry_run
        result["artifacts"] = run_logger.to_dict()
        return result

    async def _execute_bootstrap_actions(
        self,
        *,
        goal_text: str,
        actions: List[Dict[str, Any]],
        subgoal_dir: Path,
        worker: BrowserWorker,
    ) -> Dict[str, Any]:
        run_logger = MissionArtifactLogger(base_dir=subgoal_dir, goal_name=goal_text)
        worker.logger = run_logger
        prepared_actions = _ensure_screenshot_action(actions)
        result = await worker.execute({"action": prepared_actions, "goal": goal_text, "demo": self.demo_mode})
        payload = {
            "completion": result.get("completion"),
            "error": result.get("error"),
            "artifacts": run_logger.to_dict(),
        }
        if result.get("dom_snapshot"):
            payload["dom_snapshot"] = result.get("dom_snapshot")
        if result.get("secure_area"):
            payload["secure_area"] = result["secure_area"]
        return payload

    def _build_worker(self, mission_spec: MissionSpec) -> BrowserWorker:
        dry_run = not mission_spec.execute
        worker = BrowserWorker(
            settings=self.settings,
            logger=None,
            enable_playwright=False if dry_run else None,
            show_browser=None if dry_run else True,
        )
        setattr(worker, "demo_mode", self.demo_mode)
        return worker

    def _record_page_intents(
        self,
        run_ctx: Dict[str, Any],
        trace_recorder: ExecutionTraceRecorder | None,
        attempt_handle: str | None,
    ) -> None:
        if not trace_recorder:
            return
        for entry in run_ctx.get("page_intents") or []:
            trace_recorder.record_page_intent(
                intent=str(entry.get("intent") or "unknown"),
                confidence=float(entry.get("confidence", 0.0)),
                strategy=entry.get("strategy"),
                signals=entry.get("signals"),
                step_id=entry.get("step_id"),
            )

    def _extract_latest_dom_snapshot(self, payload: Dict[str, Any]) -> str | None:
        def _scan_steps(entries: List[Dict[str, Any]]) -> str | None:
            for entry in reversed(entries):
                if not isinstance(entry, dict):
                    continue
                snapshot = entry.get("dom_snapshot")
                if isinstance(snapshot, str) and snapshot.strip():
                    return snapshot
                result = entry.get("result")
                if isinstance(result, dict):
                    dom_value = result.get("dom_snapshot")
                    if isinstance(dom_value, str) and dom_value.strip():
                        return dom_value
                    nested_steps = result.get("steps")
                    if isinstance(nested_steps, list):
                        nested_dom = _scan_steps(nested_steps)
                        if nested_dom:
                            return nested_dom
            return None

        steps = payload.get("steps")
        if isinstance(steps, list):
            dom_snapshot = _scan_steps(steps)
            if dom_snapshot:
                return dom_snapshot
        fallback = payload.get("dom_snapshot")
        if isinstance(fallback, str) and fallback.strip():
            return fallback
        return None

    def _capture_page_intent_from_dom(
        self,
        *,
        run_ctx: Dict[str, Any],
        dom_snapshot: str | None,
        trace_recorder: ExecutionTraceRecorder | None,
        budget_monitor: BudgetMonitor | None = None,
    ) -> None:
        if not dom_snapshot:
            return
        if not isinstance(run_ctx, dict):
            return
        if self._resolve_page_intent(run_ctx):
            return
        url = run_ctx.get("current_url")
        result = classify_page_intent(dom_snapshot, url=url)
        strategy = self._strategy_from_intent(result.intent)
        payload = result.as_payload()
        payload.update({
            "strategy": strategy,
            "step_id": "dom_snapshot_probe",
        })
        intents = run_ctx.setdefault("page_intents", [])
        intents.append(payload)
        run_ctx["current_page_intent"] = payload
        run_ctx["page_intent"] = payload
        if result.intent is PageIntent.LISTING_PAGE and result.confidence >= 0.4:
            requests = run_ctx.setdefault("requested_skills", [])
            if not any(entry.get("name") == "listing_extraction_skill" for entry in requests):
                requests.append({
                    "name": "listing_extraction_skill",
                    "intent": result.intent.value,
                    "confidence": result.confidence,
                    "reason": "dom_snapshot_probe",
                })
        if trace_recorder:
            trace_recorder.record_page_intent(
                intent=result.intent.value,
                confidence=result.confidence,
                strategy=strategy,
                signals=payload.get("signals"),
                step_id="dom_snapshot_probe",
            )
        if budget_monitor:
            budget_monitor.record_confidence(result.confidence)
        run_ctx["_intent_probe_complete"] = True

    async def _maybe_run_listing_extraction(
        self,
        *,
        mission_spec: MissionSpec,
        worker: BrowserWorker,
        run_ctx: Dict[str, Any],
        subgoal_dir: Path,
        trace_recorder: ExecutionTraceRecorder | None,
        attempt_handle: str | None,
        learning_bias: LearningBias | None = None,
    ) -> Dict[str, Any] | None:
        if mission_spec.constraints and mission_spec.constraints.get("demo_force_actions"):
            return {"status": "skipped", "error": "listing_skill_disabled_for_demo"}
        requests = run_ctx.get("requested_skills") or []
        if not any(entry.get("name") == "listing_extraction_skill" for entry in requests):
            return None
        if not mission_spec.execute:
            if trace_recorder:
                trace_recorder.record_extraction(
                    name="listing_extraction_skill",
                    status="skipped",
                    summary={"reason": "execute_required"},
                )
            return {"status": "skipped", "error": "listing_skill_requires_execute"}
        artifact_path = subgoal_dir / "listing_extraction.json"
        skill_context = {
            "artifact_path": str(artifact_path),
            "page_url": run_ctx.get("current_url"),
        }
        skill_result = await worker.run_skill("listing_extraction_skill", skill_context)
        status = (skill_result.get("status") or "unknown").lower()
        if trace_recorder:
            trace_recorder.record_skill_usage(
                name="listing_extraction_skill",
                status=status,
                handle=attempt_handle,
                metadata={"intent": run_ctx.get("current_page_intent"), "result": skill_result},
                learning_bias=self._learning_bias_metadata(
                    learning_bias,
                    skill_name="listing_extraction_skill",
                ),
            )
            if status == "success":
                self._record_capability_usage_for_skill(
                    trace_recorder=trace_recorder,
                    handle=attempt_handle,
                    skill_id="listing_extraction_skill",
                    success=True,
                )
            trace_recorder.record_extraction(
                name="listing_extraction_skill",
                status=status,
                summary={"intent": (run_ctx.get("current_page_intent") or {}).get("intent")},
                artifact_path=str(artifact_path) if status == "success" else None,
            )
        if status != "success":
            return {"status": status, "error": skill_result.get("reason") or status}
        selected_item = skill_result.get("result") or {}
        source_url = selected_item.get("source_url") or selected_item.get("url")
        nav_artifact: Dict[str, Any] = {}
        if source_url:
            nav_result = await worker.execute({"action": {"action": "navigate", "url": source_url}, "goal": "follow_listing_result"})
            nav_artifact = {
                "followed_url": source_url,
                "result": nav_result,
            }
        artifacts = {
            "listing_extraction": {
                "path": str(artifact_path),
                "item": selected_item,
            }
        }
        if nav_artifact:
            artifacts["listing_extraction"]["follow"] = nav_artifact
        completion = {
            "complete": True,
            "reason": "listing_extraction_skill",
            "payload": {"item": selected_item},
        }
        if trace_recorder:
            trace_recorder.record_artifact("listing_extraction", str(artifact_path))
        return {"status": "success", "artifact": str(artifact_path), "item": selected_item, "completion": completion, "artifacts": artifacts}

    def _record_budget_confidence(
        self,
        run_ctx: Dict[str, Any] | None,
        budget_monitor: BudgetMonitor | None,
    ) -> None:
        if not budget_monitor or not run_ctx:
            return
        candidates: List[Dict[str, Any]] = []
        primary = run_ctx.get("current_page_intent")
        if isinstance(primary, dict):
            candidates.append(primary)
        fallback = run_ctx.get("page_intent")
        if isinstance(fallback, dict):
            candidates.append(fallback)
        intents = run_ctx.get("page_intents")
        if isinstance(intents, list) and intents:
            tail = intents[-1]
            if isinstance(tail, dict):
                candidates.append(tail)
        for payload in candidates:
            confidence = payload.get("confidence")
            if isinstance(confidence, (int, float)):
                budget_monitor.record_confidence(float(confidence))
                return

    def _record_capability_usage_for_skill(
        self,
        *,
        trace_recorder: ExecutionTraceRecorder | None,
        handle: str | None,
        skill_id: str,
        success: bool,
        confidence: float | None = None,
    ) -> None:
        if not trace_recorder or not success:
            return
        capability_ids = [cap.id for cap in capabilities_for_skill(skill_id)]
        if not capability_ids:
            return
        trace_recorder.record_capability_usage(
            skill_id=skill_id,
            handle=handle,
            capability_ids=capability_ids,
            confidence=confidence,
        )

    def _count_steps(self, payload: Dict[str, Any]) -> int:
        steps = payload.get("steps")
        if not isinstance(steps, list):
            return 0
        return self._count_step_entries(steps)

    def _count_step_entries(self, steps: List[Any]) -> int:
        count = 0
        for entry in steps:
            if not isinstance(entry, dict):
                continue
            if self._extract_action_label(entry):
                count += 1
            nested = entry.get("steps")
            if isinstance(nested, list):
                count += self._count_step_entries(nested)
            result_steps = entry.get("result", {}).get("steps") if isinstance(entry.get("result"), dict) else None
            if isinstance(result_steps, list):
                count += self._count_step_entries(result_steps)
        return count

    def _check_safety_contract(
        self,
        steps: Any,
        safety_contract: SafetyContract | None,
    ) -> Dict[str, Any] | None:
        if not safety_contract or not isinstance(steps, list):
            return None
        allowed = set((safety_contract.allowed_actions or []) or [])
        blocked = set((safety_contract.blocked_actions or []) or [])
        for action in self._iter_step_actions(steps):
            if action in blocked:
                return {
                    "status": "halted",
                    "reason": "safety_contract_blocked_action",
                    "detail": {"action": action, "policy": "blocked_actions"},
                }
            if allowed and action not in allowed:
                return {
                    "status": "ask_human",
                    "reason": "action_requires_review",
                    "detail": {"action": action, "policy": "allowed_actions"},
                }
            if safety_contract.requires_confirmation and action in CONFIRMATION_ACTIONS:
                return {
                    "status": "ask_human",
                    "reason": "confirmation_required",
                    "detail": {"action": action, "policy": "requires_confirmation"},
                }
        return None

    def _iter_step_actions(self, steps: List[Any]) -> List[str]:
        actions: List[str] = []
        stack: List[Any] = list(steps)
        while stack:
            entry = stack.pop()
            if not isinstance(entry, dict):
                continue
            label = self._extract_action_label(entry)
            if label:
                actions.append(label)
            nested = entry.get("steps")
            if isinstance(nested, list):
                stack.extend(nested)
            result = entry.get("result")
            if isinstance(result, dict):
                nested_steps = result.get("steps")
                if isinstance(nested_steps, list):
                    stack.extend(nested_steps)
        return actions

    def _extract_action_label(self, entry: Dict[str, Any]) -> str | None:
        candidates = [entry.get("action"), entry.get("name"), entry.get("type"), entry.get("operation")]
        for candidate in candidates:
            value: Any = candidate
            if isinstance(candidate, dict):
                value = candidate.get("action") or candidate.get("name")
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        return None

    def _resolve_autonomy_budget(self, mission_spec: MissionSpec) -> AutonomyBudget:
        overrides = mission_spec.autonomy_budget or {}
        return AutonomyBudget(
            max_steps=int(overrides.get("max_steps", DEFAULT_AUTONOMY_BUDGET.max_steps)),
            max_retries=int(overrides.get("max_retries", DEFAULT_AUTONOMY_BUDGET.max_retries)),
            max_duration_sec=float(overrides.get("max_duration_sec", DEFAULT_AUTONOMY_BUDGET.max_duration_sec)),
            max_risk_score=float(overrides.get("max_risk_score", DEFAULT_AUTONOMY_BUDGET.max_risk_score)),
        )

    def _resolve_safety_contract(self, mission_spec: MissionSpec) -> SafetyContract | None:
        payload = mission_spec.safety_contract
        if not isinstance(payload, dict):
            return None
        contract = SafetyContract(
            allowed_actions=self._coerce_action_list(payload.get("allowed_actions")),
            blocked_actions=self._coerce_action_list(payload.get("blocked_actions")),
            requires_confirmation=bool(payload.get("requires_confirmation", False)),
        )
        normalized = contract.normalize()
        if not normalized.allowed_actions and not normalized.blocked_actions and not normalized.requires_confirmation:
            return None
        return normalized

    def _coerce_action_list(self, value: Any) -> list[str] | None:
        if not value:
            return None
        if not isinstance(value, list):
            return None
        normalized = [str(entry).strip().lower() for entry in value if isinstance(entry, str) and entry.strip()]
        return normalized or None

    def _build_termination_payload(
        self,
        *,
        termination_type: MissionTermination,
        reason: str,
        detail: Dict[str, Any] | None,
        budget_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        status_label = termination_type.name.lower()
        return {
            "state": termination_type.value,
            "status": status_label,
            "reason": reason,
            "detail": detail or {},
            "budget_snapshot": budget_snapshot,
        }

    def _estimate_cost(self, budget_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        steps = int(budget_snapshot.get("steps_used", 0))
        retries = int(budget_snapshot.get("retries_used", 0))
        elapsed = float(budget_snapshot.get("elapsed_seconds", 0.0))
        risk = float(budget_snapshot.get("risk_score", 0.0))
        amount = round(steps * 0.018 + retries * 0.045 + elapsed * 0.002 + risk * 0.25, 4)
        return {
            "currency": "USD",
            "amount": amount,
            "basis": {
                "step_rate": 0.018,
                "retry_rate": 0.045,
                "time_rate": 0.002,
                "risk_rate": 0.25,
            },
        }

    def _build_reason_summary(
        self,
        *,
        status: MissionStatus,
        summary: Dict[str, Any],
        termination: Dict[str, Any],
    ) -> str:
        base_reason = summary.get("reason") or status
        if termination:
            detail = termination.get("detail") or {}
            code = detail.get("code") if isinstance(detail, dict) else None
            suffix = f" ({code})" if code else ""
            state = termination.get("state") or status.upper()
            return f"{state}: {base_reason}{suffix}"
        return f"{str(status).upper()}: {base_reason}"

    def _should_escalate_on_uncertainty(self, budget_monitor: BudgetMonitor) -> bool:
        average_conf = budget_monitor.usage.average_confidence()
        if average_conf is not None and average_conf < 0.4:
            return True
        if budget_monitor.usage.risk_score >= 0.9 * budget_monitor.budget.max_risk_score:
            return True
        return False

    def _maybe_skip_subgoal_for_intent(
        self,
        *,
        subgoal: MissionSubgoal,
        last_run_ctx: Dict[str, Any] | None,
        trace_recorder: ExecutionTraceRecorder | None,
    ) -> MissionSubgoalResult | None:
        if not last_run_ctx:
            return None
        intent = self._resolve_page_intent(last_run_ctx)
        if not intent:
            return None
        if intent in {PageIntent.LOGIN_FORM, PageIntent.UNKNOWN}:
            return None
        intent_label = intent.value.upper()
        if self._subgoal_is_navigation(subgoal):
            return self._build_intent_skip_result(
                subgoal=subgoal,
                intent_label=intent_label,
                reason="page_intent_known",
                trace_recorder=trace_recorder,
            )
        if self._subgoal_requires_form_logic(subgoal):
            return self._build_intent_skip_result(
                subgoal=subgoal,
                intent_label=intent_label,
                reason="page_intent_known",
                trace_recorder=trace_recorder,
            )
        return None

    def _subgoal_requires_form_logic(self, subgoal: MissionSubgoal) -> bool:
        desc = (subgoal.description or "").lower()
        return any(keyword in desc for keyword in FORM_SUBGOAL_KEYWORDS)

    def _subgoal_is_navigation(self, subgoal: MissionSubgoal) -> bool:
        metadata = subgoal.planner_metadata or {}
        bucket = str(metadata.get("bucket") or "").lower()
        desc = (subgoal.description or "").lower()
        return bucket == "navigation" or any(keyword in desc for keyword in NAV_SUBGOAL_KEYWORDS)

    def _build_intent_skip_result(
        self,
        *,
        subgoal: MissionSubgoal,
        intent_label: str,
        reason: str,
        trace_recorder: ExecutionTraceRecorder | None,
    ) -> MissionSubgoalResult:
        skip_payload = {
            "skipped_subgoal": subgoal.description,
            "reason": reason,
            "page_intent": intent_label,
        }
        if trace_recorder:
            trace_recorder.record_subgoal_skip(
                subgoal=subgoal,
                reason=reason,
                page_intent=intent_label,
            )
        now = datetime.now(UTC)
        completion = {
            "complete": True,
            "reason": reason,
            "payload": skip_payload,
            "page_intent": intent_label,
        }
        return MissionSubgoalResult(
            subgoal_id=subgoal.id,
            description=subgoal.description,
            status="skipped",
            attempts=0,
            started_at=now,
            ended_at=now,
            completion=completion,
            error=None,
            artifacts={},
        )

    def _build_strategy_violation_result(
        self,
        *,
        subgoal: MissionSubgoal,
        attempts: int,
        page_intent: str,
    ) -> MissionSubgoalResult:
        now = datetime.now(UTC)
        completion = {
            "complete": False,
            "reason": "strategy_violation",
            "page_intent": page_intent,
        }
        return MissionSubgoalResult(
            subgoal_id=subgoal.id,
            description=subgoal.description,
            status="failed",
            attempts=attempts,
            started_at=now,
            ended_at=now,
            completion=completion,
            error="strategy_violation",
            artifacts={},
        )

    def _detect_strategy_violation(
        self,
        subgoal: MissionSubgoal,
        run_ctx: Dict[str, Any] | None,
    ) -> Dict[str, Any] | None:
        intent = self._resolve_page_intent(run_ctx)
        if not intent:
            return None
        if self._subgoal_requires_form_logic(subgoal) and self._intent_blocks_form_actions(intent):
            return {
                "subgoal": subgoal.description,
                "page_intent": intent.value,
                "reason": "intent_blocks_form_action",
            }
        return None

    def _intent_blocks_form_actions(self, intent: PageIntent) -> bool:
        return intent not in SAFE_FORM_INTENTS

    def _enforce_intent_alignment(
        self,
        *,
        subgoal: MissionSubgoal,
        run_ctx: Dict[str, Any],
    ) -> None:
        intent = self._resolve_page_intent(run_ctx)
        if not intent or not self._subgoal_requires_form_logic(subgoal):
            return
        if self._intent_blocks_form_actions(intent):
            raise StrategyViolationError(subgoal=subgoal, page_intent=intent.value.upper())

    def _resolve_page_intent(self, run_ctx: Dict[str, Any] | None) -> PageIntent | None:
        if not run_ctx:
            return None
        candidates: List[Dict[str, Any]] = []
        current = run_ctx.get("current_page_intent")
        if isinstance(current, dict):
            candidates.append(current)
        primary = run_ctx.get("page_intent")
        if isinstance(primary, dict):
            candidates.append(primary)
        intents = run_ctx.get("page_intents") or []
        if intents:
            tail = intents[-1]
            if isinstance(tail, dict):
                candidates.append(tail)
        for payload in candidates:
            label = payload.get("intent")
            if not label:
                continue
            try:
                return PageIntent(str(label).lower())
            except ValueError:
                continue
        return None

    def _strategy_from_intent(self, intent: PageIntent) -> str:
        if intent is PageIntent.LOGIN_FORM:
            return "login_form"
        if intent is PageIntent.LISTING_PAGE:
            return "listing_extraction"
        if intent is PageIntent.ARTICLE_PAGE:
            return "article_extract"
        if intent is PageIntent.DASHBOARD:
            return "dashboard"
        if intent is PageIntent.DETAIL_PAGE:
            return "detail_review"
        return "unknown"

    def _build_trace_recorder(self) -> ExecutionTraceRecorder:
        if self._trace_recorder_factory:
            return self._trace_recorder_factory()
        return ExecutionTraceRecorder()

    def _finalize_trace(
        self,
        *,
        trace_recorder: ExecutionTraceRecorder,
        status: MissionStatus,
        summary: Dict[str, Any],
        end_ts: datetime,
    ) -> Path:
        trace_recorder.finalize(status=status, ended_at=end_ts)
        try:
            trace_path = trace_recorder.persist()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("execution trace persistence failed") from exc
        trace = trace_recorder.trace
        if trace is None:
            raise RuntimeError("execution trace missing after finalization")
        summary_path = write_trace_summary(trace, trace_path)
        trace_recorder.record_artifact("trace_summary", str(summary_path))
        decisions_path = write_decision_report(trace, trace_path)
        trace_recorder.record_artifact("trace_decisions", str(decisions_path))
        summary["execution_trace"] = str(trace_path)
        summary["execution_trace_summary"] = str(summary_path)
        summary["execution_trace_decisions"] = str(decisions_path)
        if trace.incomplete:
            summary["trace_incomplete"] = True
        if trace.warnings:
            summary["trace_warnings"] = list(trace.warnings)
        return trace_path

    def _bind_worker_trace(
        self,
        worker: BrowserWorker,
        trace_recorder: ExecutionTraceRecorder | None,
        attempt_handle: str | None,
    ) -> None:
        setter = getattr(worker, "set_trace_context", None)
        if callable(setter):
            setter(trace_recorder=trace_recorder, trace_handle=attempt_handle)

    def _clear_worker_trace(self, worker: BrowserWorker) -> None:
        clearer = getattr(worker, "clear_trace_context", None)
        if callable(clearer):
            clearer()

    def _write_result_file(self, mission_dir: Path, result: MissionResult) -> None:
        payload = result.model_dump(mode="json")
        (mission_dir / "mission_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _empty_capability_report(self) -> Dict[str, Any]:
        return {"required": [], "missing": [], "optional": [], "risk_level": "low"}

    def _build_capability_report(self, subgoals: List[MissionSubgoal]) -> Dict[str, Any]:
        requirements = []
        for subgoal in subgoals:
            payloads = []
            metadata = subgoal.planner_metadata or {}
            if isinstance(metadata, dict):
                payloads = metadata.get("capability_requirements") or []
            requirements.extend(requirements_from_payload(payloads))
        report = build_plan_capability_report(requirements)
        return report_to_payload(report)

    def _capability_requirements_for_subgoal(self, subgoal: MissionSubgoal) -> List[CapabilityRequirement]:
        metadata = subgoal.planner_metadata or {}
        if not isinstance(metadata, dict):
            return []
        payloads = metadata.get("capability_requirements") or []
        return requirements_from_payload(payloads)

    def _predicted_actions_for_subgoal(self, subgoal: MissionSubgoal) -> List[str]:
        metadata = subgoal.planner_metadata or {}
        if isinstance(metadata, dict):
            actions = metadata.get("predicted_actions") or metadata.get("actions") or []
            if isinstance(actions, list):
                return [str(action) for action in actions]
        return []

    def _evaluate_judgment(
        self,
        *,
        mission_spec: MissionSpec,
        subgoal: MissionSubgoal,
        capability_requirements: List[CapabilityRequirement],
        safety_contract: SafetyContract | None,
        learning_bias: LearningBias | None,
        last_page_intent: str | None,
    ) -> "JudgmentDecision":
        predicted_actions = self._predicted_actions_for_subgoal(subgoal)
        return self._judgment_evaluator.evaluate(
            mission_spec=mission_spec,
            subgoal=subgoal,
            capability_requirements=capability_requirements,
            safety_contract=safety_contract,
            learning_bias=learning_bias,
            predicted_actions=predicted_actions,
            page_intent=last_page_intent,
        )

    def _capability_enforcement_context(self) -> EnforcementContext:
        cfg = self._capability_enforcement_cfg or {}
        threshold = float(cfg.get("threshold", 0.8) or 0.8)
        critical = float(cfg.get("critical", 0.5) or 0.5)
        threshold = min(1.0, max(0.0, threshold))
        critical = min(threshold, max(0.0, critical))
        return EnforcementContext(
            threshold=threshold,
            critical=critical,
            auto_approve_capabilities=bool(cfg.get("auto_approve_capabilities", False)),
            fail_on_missing_capability=bool(cfg.get("fail_on_missing_capability", False)),
        )

    def _evaluate_capability_enforcements(
        self,
        *,
        requirements: List[CapabilityRequirement],
        subgoal_id: str,
    ) -> List[CapabilityDecision]:
        if not requirements:
            return []
        context = self._capability_enforcement_context()
        decisions = evaluate_capabilities(requirements, context, registry=CAPABILITY_REGISTRY)
        return [replace(decision, subgoal_id=subgoal_id) for decision in decisions]

    def _to_trace_capability_enforcements(
        self, decisions: List[CapabilityDecision]
    ) -> List[CapabilityEnforcementDecision]:
        trace_decisions: List[CapabilityEnforcementDecision] = []
        for decision in decisions:
            trace_decisions.append(
                CapabilityEnforcementDecision(
                    capability_id=decision.capability_id,
                    decision=decision.decision,
                    confidence=decision.confidence,
                    threshold=decision.threshold,
                    critical=decision.critical,
                    reason=decision.reason,
                    required=decision.required,
                    missing=decision.missing,
                    subgoal_id=decision.subgoal_id,
                    source=decision.source,
                )
            )
        return trace_decisions

    def _should_request_approval(
        self,
        *,
        capability_enforcements: List[CapabilityDecision],
        mission_spec: MissionSpec,
        subgoal: MissionSubgoal,
        learning_bias: LearningBias | None,
    ) -> tuple[bool, str, str]:
        cfg = self._approval_cfg or {}
        if cfg.get("require_approval"):
            return True, "require_approval_flag", "medium"
        if any(decision.decision == "ask_human" for decision in capability_enforcements):
            risk = "high" if any(dec.missing for dec in capability_enforcements) else "medium"
            return True, "capability_enforcement", risk
        metadata = subgoal.planner_metadata or {}
        bucket = str(metadata.get("bucket") or "").lower()
        if bucket in {"listing_extraction", "data_extraction"} or "data extraction" in subgoal.description.lower():
            return True, "data_extraction", "medium"
        if "file" in subgoal.description.lower():
            return True, "file_write", "medium"
        if "api" in subgoal.description.lower():
            return True, "external_api", "high"
        if learning_bias and getattr(learning_bias, "confidence", None) is not None:
            try:
                if float(getattr(learning_bias, "confidence")) < 0.25:
                    return True, "learning_bias_low_confidence", "medium"
            except Exception:
                pass
        return False, "", "low"

    async def _maybe_request_approval(
        self,
        *,
        mission_dir: Path,
        mission_spec: MissionSpec,
        subgoal: MissionSubgoal,
        capability_requirements: List[CapabilityRequirement],
        capability_enforcements: List[CapabilityDecision],
        trace_recorder: ExecutionTraceRecorder | None,
        learning_bias: LearningBias | None,
    ) -> tuple[ApprovalState, str | None, Path | None]:
        triggered, reason, risk_level = self._should_request_approval(
            capability_enforcements=capability_enforcements,
            mission_spec=mission_spec,
            subgoal=subgoal,
            learning_bias=learning_bias,
        )
        if not triggered:
            return "not_required", None, None
        timeout_secs = int(self._approval_cfg.get("timeout_secs", 300)) if isinstance(self._approval_cfg, dict) else 300
        approval_request = self._build_approval_request_payload(
            mission_spec=mission_spec,
            subgoal=subgoal,
            capability_requirements=capability_requirements,
            reason=reason,
            risk_level=risk_level,
            timeout_secs=timeout_secs,
        )
        approval_path = self._write_approval_request(mission_dir=mission_dir, request=approval_request)
        if approval_path and trace_recorder:
            trace_recorder.record_artifact("approval_request", str(approval_path))
            trace_recorder.record_approval_request(
                approval_id=approval_request.approval_id,
                subgoal_id=subgoal.id,
                reason=reason,
                risk_level=risk_level,
                requested_action=approval_request.requested_action,
                expires_at=approval_request.expires_at,
            )
        state, resolved_reason = await self._await_approval_decision(
            request=approval_request,
            path=approval_path or (mission_dir / "approval_request.json"),
            trace_recorder=trace_recorder,
            mission_spec=mission_spec,
        )
        return state, resolved_reason, approval_path

    def _build_approval_request_payload(
        self,
        *,
        mission_spec: MissionSpec,
        subgoal: MissionSubgoal,
        capability_requirements: List[CapabilityRequirement],
        reason: str,
        risk_level: str,
        timeout_secs: int,
    ) -> ApprovalRequest:
        approval_id = f"apr_{datetime.now(APPROVAL_UTC).strftime('%Y%m%d_%H%M%S')}"
        metadata = subgoal.planner_metadata or {}
        requested_action = {
            "type": "skill" if metadata.get("skill") else "subgoal",
            "name": metadata.get("skill") or subgoal.description,
            "target": metadata.get("primary_url") or self._detect_primary_url([subgoal]) or "unknown",
        }
        expires_at = datetime.now(APPROVAL_UTC) + timedelta(seconds=max(5, timeout_secs))
        capabilities_required = [req.capability_id for req in capability_requirements]
        learning_payload: Dict[str, Any] = {}
        if metadata.get("learning_override"):
            learning_payload.update(metadata.get("learning_override") or {})
        alternatives = ["Manual confirmation", "Skip action", "Abort mission"]
        return ApprovalRequest(
            approval_id=approval_id,
            mission_id=mission_spec.id,
            subgoal_id=subgoal.id,
            requested_action=requested_action,
            reason=reason,
            risk_level=risk_level,
            capabilities_required=capabilities_required,
            learning_bias=learning_payload,
            alternatives=alternatives,
            expires_at=expires_at,
        )

    def _write_approval_request(self, *, mission_dir: Path, request: ApprovalRequest) -> Path | None:
        try:
            path = mission_dir / "approval_request.json"
            path.write_text(json.dumps(request.to_payload(), indent=2), encoding="utf-8")
            return path
        except Exception:
            self.logger.warning("failed to write approval_request.json", exc_info=True)
            return None

    def _read_approval_request(self, path: Path) -> ApprovalRequest | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return ApprovalRequest.from_payload(payload)
        except Exception:
            return None

    def _persist_approval_resolution(
        self,
        *,
        request: ApprovalRequest,
        state: ApprovalState,
        reason: str | None,
        resolved_by: str | None,
        path: Path,
    ) -> None:
        request.state = state
        request.resolved_at = datetime.now(APPROVAL_UTC)
        request.resolved_by = resolved_by
        request.resolution_reason = reason
        if state == "approved":
            request.approved_by_human = True
        path.write_text(json.dumps(request.to_payload(), indent=2), encoding="utf-8")

    async def _await_approval_decision(
        self,
        *,
        request: ApprovalRequest,
        path: Path,
        trace_recorder: ExecutionTraceRecorder | None,
        mission_spec: MissionSpec,
    ) -> tuple[ApprovalState, str | None]:
        cfg = self._approval_cfg or {}
        if cfg.get("auto_approve_low_risk") and request.risk_level in {"low", "medium"}:
            self._persist_approval_resolution(
                request=request,
                state="approved",
                reason="auto_approved_low_risk",
                resolved_by="auto",
                path=path,
            )
            if trace_recorder:
                trace_recorder.record_approval_resolution(
                    approval_id=request.approval_id,
                    subgoal_id=request.subgoal_id,
                    state="approved",
                    resolved_by="auto",
                    reason="auto_approved_low_risk",
                    external=False,
                )
            return "approved", "auto_approved_low_risk"

        timeout = max(5, int(cfg.get("timeout_secs", 300)))
        deadline = datetime.now(APPROVAL_UTC) + timedelta(seconds=timeout)
        while datetime.now(APPROVAL_UTC) < deadline:
            await self._sleep(1.0)
            latest = self._read_approval_request(path)
            if latest and latest.state in {"approved", "rejected", "expired"}:
                state = latest.state
                reason = latest.resolution_reason or latest.reason
                if trace_recorder:
                    trace_recorder.record_approval_resolution(
                        approval_id=latest.approval_id,
                        subgoal_id=latest.subgoal_id,
                        state=state,
                        resolved_by=latest.resolved_by,
                        reason=reason,
                        external=True,
                    )
                return state, reason
            if latest and latest.expires_at < datetime.now(APPROVAL_UTC):
                self._persist_approval_resolution(
                    request=latest,
                    state="expired",
                    reason="approval_timeout",
                    resolved_by="timeout",
                    path=path,
                )
                if trace_recorder:
                    trace_recorder.record_approval_resolution(
                        approval_id=latest.approval_id,
                        subgoal_id=latest.subgoal_id,
                        state="expired",
                        resolved_by="timeout",
                        reason="approval_timeout",
                        external=False,
                    )
                return "expired", "approval_timeout"
        self._persist_approval_resolution(
            request=request,
            state="expired",
            reason="approval_timeout",
            resolved_by="timeout",
            path=path,
        )
        if trace_recorder:
            trace_recorder.record_approval_resolution(
                approval_id=request.approval_id,
                subgoal_id=request.subgoal_id,
                state="expired",
                resolved_by="timeout",
                reason="approval_timeout",
                external=False,
            )
        return "expired", "approval_timeout"

    def _load_capability_enforcement_settings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        threshold = float(payload.get("threshold", 0.8) or 0.8)
        critical = float(payload.get("critical", 0.5) or 0.5)
        threshold = min(1.0, max(0.0, threshold))
        critical = min(threshold, max(0.0, critical))
        config = {
            "threshold": threshold,
            "critical": critical,
            "auto_approve_capabilities": bool(payload.get("auto_approve_capabilities", False)),
            "fail_on_missing_capability": bool(payload.get("fail_on_missing_capability", False)),
        }
        return config

    def _load_approval_settings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "require_approval": bool(payload.get("require_approval", False)),
            "timeout_secs": int(payload.get("timeout_secs", payload.get("approval_timeout", 300)) or 300),
            "auto_approve_low_risk": bool(payload.get("auto_approve_low_risk", False)),
        }

    def _write_capability_report(self, *, mission_dir: Path, capability_report: Dict[str, Any]) -> Path | None:
        try:
            payload = capability_report or self._empty_capability_report()
            path = mission_dir / "capability_report.json"
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return path
        except Exception:
            self.logger.warning("failed to write capability_report.json", exc_info=True)
            return None

    def _attach_capability_report(self, summary: Dict[str, Any], capability_report: Dict[str, Any], capability_report_path: Path | None) -> None:
        summary.setdefault("capability_report", capability_report or self._empty_capability_report())
        if capability_report_path:
            summary.setdefault("capability_report_path", str(capability_report_path))

    def _attach_capability_enforcement(self, summary: Dict[str, Any], capability_enforcement_path: Path | None) -> None:
        summary.setdefault("capability_enforcement", [decision.as_dict() for decision in self._capability_decisions])
        if capability_enforcement_path:
            summary.setdefault("capability_enforcement_path", str(capability_enforcement_path))

    def _attach_approval_requests(self, summary: Dict[str, Any]) -> None:
        if self._approval_request_paths:
            summary.setdefault("approval_requests", list(self._approval_request_paths))

    def _persist_judgment_artifacts(
        self,
        *,
        mission_dir: Path,
        decision: "JudgmentDecision",
        subgoal: MissionSubgoal,
        trace_recorder: ExecutionTraceRecorder | None,
    ) -> Dict[str, str]:
        payload = {
            "subgoal_id": subgoal.id,
            "subgoal_description": subgoal.description,
            **decision.to_payload(),
        }
        decision_path = mission_dir / "judgment_decision.json"
        explanation_path = mission_dir / "decision_explanation.json"
        decision_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        explanation = {
            "what_detected": decision.risk_factors,
            "why_risky": decision.explanation,
            "what_if_continue": "Mission would proceed with potential irreversible or authority-bound action",
        }
        explanation_path.write_text(json.dumps(explanation, indent=2), encoding="utf-8")
        if trace_recorder:
            trace_recorder.record_artifact("judgment_decision", str(decision_path))
            trace_recorder.record_artifact("decision_explanation", str(explanation_path))
        return {
            "judgment_decision_path": str(decision_path),
            "decision_explanation_path": str(explanation_path),
        }

    def _emit_judgment_approval_request(
        self,
        *,
        mission_dir: Path,
        mission_spec: MissionSpec,
        subgoal: MissionSubgoal,
        decision: "JudgmentDecision",
        capability_requirements: List[CapabilityRequirement],
        trace_recorder: ExecutionTraceRecorder | None,
    ) -> Dict[str, Any]:
        approval_request = self._build_approval_request_payload(
            mission_spec=mission_spec,
            subgoal=subgoal,
            capability_requirements=capability_requirements,
            reason=decision.explanation,
            risk_level="high",
            timeout_secs=int(self._approval_cfg.get("timeout_secs", 300)) if isinstance(self._approval_cfg, dict) else 300,
        )
        approval_path = self._write_approval_request(mission_dir=mission_dir, request=approval_request)
        if approval_path:
            self._approval_request_paths.append(str(approval_path))
            if trace_recorder:
                trace_recorder.record_artifact("approval_request", str(approval_path))
                trace_recorder.record_approval_request(
                    approval_id=approval_request.approval_id,
                    subgoal_id=subgoal.id,
                    reason=decision.explanation,
                    risk_level="high",
                    requested_action=approval_request.requested_action,
                    expires_at=approval_request.expires_at,
                )
        return {"approval_request_path": str(approval_path) if approval_path else None, "approval_requests": list(self._approval_request_paths)}

    def _persist_capability_enforcement(
        self,
        *,
        mission_dir: Path,
        mission_id: str,
        trace_recorder: ExecutionTraceRecorder | None,
        summary: Dict[str, Any],
    ) -> Path | None:
        enforcement_path = self._write_capability_enforcement(mission_dir=mission_dir, mission_id=mission_id)
        if enforcement_path and trace_recorder:
            trace_recorder.record_artifact("capability_enforcement", str(enforcement_path))
        self._attach_capability_enforcement(summary, enforcement_path)
        return enforcement_path

    def _write_capability_enforcement(self, *, mission_dir: Path, mission_id: str) -> Path | None:
        try:
            decisions = sorted(
                self._capability_decisions,
                key=lambda decision: (
                    decision.subgoal_id or "",
                    decision.capability_id,
                    decision.decision,
                ),
            )
            payload = {
                "mission_id": mission_id,
                "config": dict(self._capability_enforcement_cfg or {}),
                "decisions": [decision.as_dict() for decision in decisions],
            }
            path = mission_dir / "capability_enforcement.json"
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return path
        except Exception:
            self.logger.warning("failed to write capability_enforcement.json", exc_info=True)
            return None

    def _record_memory(self, summary: Dict[str, Any]) -> None:
        if not self.memory_manager:
            return
        add_memory = getattr(self.memory_manager, "add_memory", None)
        if callable(add_memory):
            try:
                add_memory(summary)
            except Exception:  # pragma: no cover - defensive logging
                self.logger.warning("mission memory write failed", exc_info=True)

    def _compute_learning_score(self, step: Dict[str, Any]) -> float:
        intent = step.get("intent") or step.get("bucket") or "unknown"
        return self._impact_scorer.score(step.get("skill"), step.get("description", ""), intent)

    def _persist_learning_index(self) -> None:
        index_path = self._learning_logs_root / "learning_index.json"
        try:
            self._impact_scorer.persist(index_path)
        except Exception:  # pragma: no cover - defensive logging
            self.logger.debug("failed to persist learning index", exc_info=True)

    def _detect_conflicts(self, plan_steps: List[Dict[str, Any]]) -> List[PlannerConflict]:
        conflicts: List[PlannerConflict] = []
        for step in plan_steps:
            score = float(step.get("learning_score", 0.0))
            if score <= self._learning_threshold:
                conflicts.append(
                    PlannerConflict(
                        planner_step=step,
                        learning_score=score,
                        historical_failures=max(0, int(step.get("failures", 0))),
                        recommendation="replace_or_skip" if score < 0 else "review",
                    )
                )
        return conflicts

    def _apply_adjusted_plan(self, original: List[MissionSubgoal], adjusted: Optional[List[Dict[str, Any]]]) -> List[MissionSubgoal]:
        if not adjusted:
            return original
        by_id = {subgoal.id: subgoal for subgoal in original}
        new_plan: List[MissionSubgoal] = []
        for step in adjusted:
            sg_id = step.get("id")
            if sg_id and sg_id in by_id:
                subgoal = by_id[sg_id]
            else:
                subgoal = MissionSubgoal(id=sg_id or step.get("description", "override"), description=step.get("description", "override"), planner_metadata={})
            meta = dict(subgoal.planner_metadata or {})
            meta["learning_override"] = {
                "learning_score": step.get("learning_score"),
                "decision": step.get("decision"),
                "skill": step.get("skill"),
            }
            subgoal.planner_metadata = meta
            new_plan.append(subgoal)
        return new_plan

    def _to_plan_steps(self, subgoals: List[MissionSubgoal]) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        for subgoal in subgoals:
            metadata = subgoal.planner_metadata or {}
            steps.append(
                {
                    "id": subgoal.id,
                    "description": subgoal.description,
                    "intent": metadata.get("bucket") or metadata.get("intent") or "unknown",
                    "skill": metadata.get("skill"),
                }
            )
        return steps

    def _learning_review(
        self,
        *,
        mission_spec: MissionSpec,
        subgoals: List[MissionSubgoal],
        learning_bias: LearningBias | None,
        trace_recorder: ExecutionTraceRecorder,
    ) -> tuple[List[MissionSubgoal], Optional[Dict[str, Any]], Dict[str, Any]]:
        steps = self._to_plan_steps(subgoals)
        scores: Dict[Tuple[str, str, str], float] = {}
        for step in steps:
            score = self._compute_learning_score(step)
            step["learning_score"] = score
            key = (step.get("skill") or "unknown", step.get("description", ""), step.get("intent") or "unknown")
            scores[key] = score
            if trace_recorder:
                trace_recorder.record_learning_event(
                    event="learning_score_snapshot",
                    data={"step": step, "score": score},
                )
        self._persist_learning_index()
        min_score = min(scores.values()) if scores else 0.0
        conflicts = self._detect_conflicts(steps)
        context: Dict[str, Any] = {
            "min_score": min_score,
            "scores": scores,
            "conflicts": conflicts,
            "target_skills": [step.get("skill") or "unknown" for step in steps],
        }
        if conflicts and any(conf.learning_score <= self._learning_hard_floor for conf in conflicts):
            payload = {
                "status": "refused_by_learning",
                "summary": {
                    "reason": "learning_refusal",
                    "detail": {
                        "conflicts": [conf.learning_score for conf in conflicts],
                        "hard_floor": self._learning_hard_floor,
                    },
                },
            }
            context.update({"decision_type": "REFUSE", "reason": "learning_refusal", "confidence": abs(min_score)})
            trace_recorder.record_learning_event(
                event="planner_conflict_detected",
                data={"conflicts": [conf.__dict__ for conf in conflicts], "decision": "refuse"},
            )
            return subgoals, payload, context

        override_engine = LearningOverrideEngine(
            scores=scores,
            preferred_skills=learning_bias.preferred_skills if learning_bias else [],
            threshold=self._learning_threshold,
            hard_floor=self._learning_hard_floor,
        )
        decision = override_engine.apply_override(steps, {"learning_bias": learning_bias.as_metadata() if learning_bias else None})
        context.update(
            {
                "decision_type": decision.decision_type,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "evidence": decision.evidence,
                "adjusted_plan": decision.adjusted_plan,
            }
        )
        trace_recorder.record_learning_event(
            event="learning_override_applied" if decision.decision_type != "ACCEPT" else "learning_override_review",
            data={
                "decision": decision.decision_type,
                "reason": decision.reason,
                "evidence": decision.evidence,
            },
        )
        if decision.decision_type == "REFUSE":
            payload = {
                "status": "refused_by_learning",
                "summary": {
                    "reason": "learning_refusal",
                    "detail": {"decision": decision.reason},
                },
            }
            return subgoals, payload, context
        adjusted = self._apply_adjusted_plan(subgoals, decision.adjusted_plan)
        return adjusted, None, context

    def _load_learning_signals(self) -> List[SkillSignal]:
        try:
            return load_skill_signals(
                self._learning_logs_root,
                min_confidence=self._learning_index_cache.min_confidence,
            )
        except Exception:  # pragma: no cover - defensive logging
            self.logger.debug("learning signals read failed", exc_info=True)
            return []

    def _emit_learning_artifacts(
        self,
        *,
        mission_id: str,
        mission_dir: Path,
        before_signals: Sequence[SkillSignal],
        after_signals: Sequence[SkillSignal],
    ) -> None:
        if not mission_id:
            return
        if not mission_dir.exists():
            mission_dir.mkdir(parents=True, exist_ok=True)
        emit_learning_artifacts(
            mission_id=mission_id,
            mission_dir=mission_dir,
            before_signals=before_signals,
            after_signals=after_signals,
        )

    def _learning_triggering_signals(
        self,
        *,
        learning_bias: LearningBias | None,
        target_skills: List[str] | None,
        evidence: str,
    ) -> List[Dict[str, Any]]:
        catalog: Dict[str, SkillSignal] = {}
        for signal in getattr(self._impact_scorer, "signals", []):
            catalog.setdefault(signal.skill_name, signal)
        if learning_bias:
            for signal in learning_bias.signals:
                catalog.setdefault(signal.skill_name, signal)
        skills = [skill for skill in (target_skills or []) if skill]
        if not skills and catalog:
            skills = list(catalog.keys())
        signals: List[Dict[str, Any]] = []
        for skill in skills[:3]:
            signal = catalog.get(skill)
            failures = (signal.attempts - signal.successes) if signal else 0
            success_rate = signal.success_rate if signal else 0.0
            signals.append(
                {
                    "skill": skill,
                    "success_rate": round(success_rate, 4),
                    "failures": int(failures),
                    "evidence": evidence,
                }
            )
        if not signals:
            signals.append({"skill": "unknown", "success_rate": 0.0, "failures": 0, "evidence": evidence})
        return signals

    def _learning_override_applied(self, context: Dict[str, Any] | None) -> bool:
        if not context:
            return False
        decision_type = str(context.get("decision_type") or "").upper()
        if decision_type == "REFUSE":
            return False
        return bool(decision_type and decision_type != "ACCEPT")

    def _learning_altered_execution(
        self,
        *,
        learning_bias: LearningBias | None,
        learning_context: Dict[str, Any] | None,
        baseline_skill_plan: List[str],
        merged_skill_plan: List[str],
    ) -> bool:
        if learning_context and str(learning_context.get("decision_type") or "").upper() == "REFUSE":
            return True
        if self._learning_override_applied(learning_context):
            return True
        if learning_bias:
            return True
        return False

    def _render_learning_summary(
        self,
        *,
        decision_type: str,
        learning_context: Dict[str, Any] | None,
        learning_bias: LearningBias | None,
    ) -> str:
        impact_score = float(learning_context.get("min_score", 0.0)) if learning_context else 0.0
        reason = (learning_context.get("reason") if learning_context else None) or decision_type
        if decision_type == "refusal":
            return f"Agent refused the mission because learning score {impact_score:.2f} triggered {reason}."
        if decision_type == "override":
            return f"Learning override applied ({reason}); impact_score={impact_score:.2f}."
        bias_detail = ", ".join(learning_bias.preferred_skills) if learning_bias and learning_bias.preferred_skills else "learning_bias"
        return f"Learning bias applied to preferred skills: {bias_detail}; impact_score={impact_score:.2f}."

    def _build_learning_explanation(
        self,
        *,
        mission_id: str,
        decision_type: str,
        final_resolution: str,
        learning_context: Dict[str, Any] | None,
        learning_bias: LearningBias | None,
        summary_text: str,
        target_skills: List[str] | None = None,
    ) -> LearningDecisionExplanation:
        impact_score = float(learning_context.get("min_score", 0.0)) if learning_context else 0.0
        confidence = float(learning_context.get("confidence", 0.0) if learning_context else 0.0)
        if not confidence and learning_bias and learning_bias.signals:
            confidence = learning_bias.signals[0].confidence_mean
        evidence_label = (learning_context.get("reason") if learning_context else decision_type) or decision_type
        signals = self._learning_triggering_signals(
            learning_bias=learning_bias,
            target_skills=target_skills or (learning_context.get("target_skills") if learning_context else None),
            evidence=evidence_label,
        )
        return build_learning_decision_explanation(
            mission_id=mission_id,
            decision_type=decision_type,
            learning_impact_score=impact_score,
            confidence_score=confidence,
            triggering_signals=signals,
            planner_conflict=bool(learning_context and learning_context.get("conflicts")),
            final_resolution=final_resolution,
            summary=summary_text,
        )

    def _emit_learning_explanation(
        self,
        *,
        mission_dir: Path,
        trace_recorder: ExecutionTraceRecorder | None,
        explanation: LearningDecisionExplanation,
    ) -> Path:
        path = write_learning_decision_explanation(mission_dir, explanation)
        if trace_recorder:
            trace_recorder.record_artifact("learning_decision_explanation", str(path))
        return path

    def _record_learning(
        self,
        *,
        mission_result_path: Path,
        mission_instruction: str | None,
        trace_path: Path | None,
        mission_exception: Exception | None,
    ) -> None:
        if not mission_result_path.exists():
            return
        mission_dir = mission_result_path.parent
        mission_data: Dict[str, Any] = {}
        mission_id = mission_result_path.stem
        force_failure = False
        try:
            mission_data = json.loads(mission_result_path.read_text(encoding="utf-8"))
            mission_id = str(mission_data.get("mission_id") or mission_data.get("mission") or mission_id)
            status_text = str(mission_data.get("status", "")).lower()
            force_failure = status_text != "complete" or mission_exception is not None
        except Exception:  # pragma: no cover - defensive logging
            self.logger.warning("learning mission_result read failed", exc_info=True)
            force_failure = True

        before_signals = self._load_learning_signals()
        try:
            self.learning_recorder.record(
                mission_result_path=mission_result_path,
                trace_path=trace_path,
                mission_instruction=mission_instruction,
                runtime_error=mission_exception,
                force_outcome_failure=force_failure,
                skill_summary=None,
            )
        except Exception:  # pragma: no cover - defensive logging
            self.logger.warning("learning record write failed", exc_info=True)

        try:
            after_signals = self._load_learning_signals()
            self._emit_learning_artifacts(
                mission_id=mission_id,
                mission_dir=mission_dir,
                before_signals=before_signals,
                after_signals=after_signals,
            )
        except Exception:  # pragma: no cover - defensive logging
            self.logger.warning("learning diff artifact generation failed", exc_info=True)

    def _resolve_learning_bias(self, mission_spec: MissionSpec) -> LearningBias | None:
        if not self._learning_bias_enabled or not mission_spec.execute:
            return None
        try:
            return self._learning_index_cache.bias_for_goal(mission_spec.instruction)
        except Exception:  # pragma: no cover - cache failures are non-fatal
            self.logger.debug("learning bias lookup failed", exc_info=True)
            return None

    def _merge_skill_plan(self, skill_plan: List[str] | None, learning_bias: LearningBias | None) -> List[str]:
        merged: List[str] = list(skill_plan or [])
        if not learning_bias:
            return merged
        for skill in learning_bias.preferred_skills:
            if skill not in merged:
                merged.append(skill)
        return merged

    def _learning_bias_metadata(
        self,
        learning_bias: LearningBias | None,
        *,
        skill_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not learning_bias:
            return None
        if skill_name:
            payload = learning_bias.metadata_for(skill_name)
            if payload and context:
                payload = {**payload, "context": dict(context)}
            return payload
        return learning_bias.as_metadata(context)

    def _bind_worker_learning_bias(self, worker: BrowserWorker, learning_bias: LearningBias | None) -> None:
        setter = getattr(worker, "set_learning_bias", None)
        if callable(setter):
            metadata = None
            if learning_bias:
                metadata = learning_bias.as_metadata({"scope": "mission"})
            setter(metadata)

    def _persist_mission_memory(
        self,
        *,
        mission_result: MissionResult,
        mission_spec: MissionSpec,
        detected_url: str | None,
        used_skills: List[str],
        artifacts_dir: Path,
    ) -> None:
        try:
            memory = MissionMemory(
                mission_id=mission_result.mission_id,
                mission_text=mission_spec.instruction,
                url=detected_url,
                status=mission_result.status,
                skills_used=list(used_skills or []),
                artifacts_path=str(artifacts_dir),
                timestamp=datetime.utcnow().isoformat(),
            )
            save_mission_memory(memory)
        except Exception:  # pragma: no cover - persistence best effort
            self.logger.warning("mission memory persistence failed", exc_info=True)

    def _plan_subgoals(self, mission_spec: MissionSpec) -> Tuple[List[MissionSubgoal], Dict[str, Any]]:
        try:
            subgoals = mission_planner_module.plan_mission(mission_spec, settings=self.settings)
        except TypeError as exc:
            if "unexpected keyword argument 'settings'" in str(exc):
                subgoals = mission_planner_module.plan_mission(mission_spec)
            else:
                raise
        capability_report = self._build_capability_report(subgoals)
        return subgoals, capability_report

    def _detect_primary_url(self, subgoals: List[MissionSubgoal]) -> str | None:
        for subgoal in subgoals:
            metadata = subgoal.planner_metadata or {}
            url = metadata.get("primary_url")
            if url:
                return url
            description = subgoal.description.lower()
            if "navigate to" in description and "http" in description:
                parts = description.split("navigate to", 1)[-1].strip()
                if parts:
                    return parts.split()[0]
        return None

    def _should_apply_login_skill(
        self,
        subgoal: MissionSubgoal,
        mission_spec: MissionSpec,
        skill_plan: List[str] | None,
        used_skills: List[str] | None,
    ) -> bool:
        constraints = mission_spec.constraints or {}
        if isinstance(constraints, dict) and constraints.get("disable_login_skill"):
            return False
        if not mission_spec.execute:
            return False
        if not skill_plan or "login_form_skill" not in skill_plan:
            return False
        if used_skills and "login_form_skill" in used_skills:
            return False
        description = subgoal.description.lower()
        prefix = description.split(":", 1)[0]
        return "login" in prefix

    async def _invoke_login_skill(self, *, worker: BrowserWorker, mission_spec: MissionSpec, url: str | None) -> Dict[str, Any]:
        context = self._build_skill_context(mission_spec, url)
        result = await worker.run_skill("login_form_skill", context)
        return {"skill": "login_form_skill", "result": result}

    def _build_skill_context(self, mission_spec: MissionSpec, url: str | None) -> Dict[str, Any]:
        constraints = mission_spec.constraints or {}
        cred_source = constraints.get("credentials") or {}
        username = cred_source.get("username")
        password = cred_source.get("password")
        instruction_lower = mission_spec.instruction.lower()
        if not username:
            if "tomsmith" in instruction_lower:
                username = "tomsmith"
            elif "student" in instruction_lower:
                username = "student"
        if not password:
            if "supersecretpassword" in instruction_lower:
                password = "SuperSecretPassword!"
            elif "password123" in instruction_lower:
                password = "Password123"
        username = username or os.getenv("TUSK_DEFAULT_USERNAME", "student")
        password = password or os.getenv("TUSK_DEFAULT_PASSWORD", "Password123")
        payload: Dict[str, Any] = {"username": username, "password": password}
        if url:
            payload["url"] = url
        return payload

    def _predict_trace_path(self, trace_recorder: ExecutionTraceRecorder) -> Path:
        trace = trace_recorder.trace
        if not trace:
            raise RuntimeError("execution trace not initialized")
        return trace_recorder.serializer.build_path(trace, directory=trace_recorder.storage_dir)

    def _persist_resume_checkpoint(
        self,
        *,
        mission_dir: Path,
        checkpoint: ResumeCheckpoint,
        trace_recorder: ExecutionTraceRecorder | None,
    ) -> Path:
        path = mission_dir / "resume_checkpoint.json"
        try:
            checkpoint.save(path)
            if trace_recorder:
                trace_recorder.record_artifact("resume_checkpoint", str(path))
            return path
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("resume checkpoint persistence failed") from exc

    def _checkpoint_path_from_resume(self, resume_from: str | Path) -> Path:
        candidate = Path(resume_from)
        if candidate.exists():
            return candidate
        # treat resume_from as mission_id
        mission_dirs = sorted(self.artifacts_root.glob(f"mission_*_{resume_from}*"), reverse=True)
        for mission_dir in mission_dirs:
            checkpoint = mission_dir / "resume_checkpoint.json"
            if checkpoint.exists():
                return checkpoint
        raise FileNotFoundError(f"Resume checkpoint not found for {resume_from}")

    def _load_resume_checkpoint(self, resume_from: str | Path) -> ResumeCheckpoint:
        path = self._checkpoint_path_from_resume(resume_from)
        return ResumeCheckpoint.load(path)

    def _load_prior_subgoal_results(self, mission_dir: Path) -> List[MissionSubgoalResult]:
        results_path = mission_dir / "mission_result.json"
        if not results_path.exists():
            return []
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            items = payload.get("subgoal_results") or []
            return [MissionSubgoalResult.model_validate(item) for item in items if isinstance(item, dict)]
        except Exception:
            return []

    def _select_subgoals_for_resume(self, subgoals: List[MissionSubgoal], checkpoint: ResumeCheckpoint) -> List[MissionSubgoal]:
        if not checkpoint.pending_subgoals:
            return subgoals
        pending = set(checkpoint.pending_subgoals)
        return [sg for sg in subgoals if sg.id in pending]

    def _load_mission_text_from_trace(self, trace_path: str | None) -> str | None:
        if not trace_path:
            return None
        try:
            trace_payload = json.loads(Path(trace_path).read_text(encoding="utf-8"))
            return str(trace_payload.get("mission_text")) if isinstance(trace_payload, dict) else None
        except Exception:
            return None

    def _compute_escalation_budget(self, base_budget: AutonomyBudget) -> Dict[str, Any]:
        return {
            "max_steps": base_budget.max_steps + ESCALATION_STEP_BONUS,
            "max_retries": base_budget.max_retries,
            "max_duration_sec": base_budget.max_duration_sec + ESCALATION_TIME_BONUS_SEC,
            "max_risk_score": base_budget.max_risk_score + ESCALATION_RISK_BONUS,
        }

    def _enter_escalation(
        self,
        *,
        mission_dir: Path,
        trace_recorder: ExecutionTraceRecorder | None,
        escalation_state: EscalationState,
        budget_monitor: BudgetMonitor,
        base_budget: AutonomyBudget,
        expanded_budget: Dict[str, Any],
        window_limits: Dict[str, Any],
        limit_detail: Dict[str, Any],
    ) -> Dict[str, str]:
        artifacts: Dict[str, str] = {}
        escalation_state.mark_requested("risk_budget_exceeded", expanded_budget=expanded_budget, window_limits=window_limits)
        escalation_state.allowed = False
        budget_monitor.budget = AutonomyBudget(
            max_steps=int(expanded_budget.get("max_steps", base_budget.max_steps)),
            max_retries=int(expanded_budget.get("max_retries", base_budget.max_retries)),
            max_duration_sec=float(expanded_budget.get("max_duration_sec", base_budget.max_duration_sec)),
            max_risk_score=float(expanded_budget.get("max_risk_score", base_budget.max_risk_score)),
        )
        request_payload = {
            "reason": "risk_budget_exceeded",
            "detail": limit_detail,
            "requested_at": escalation_state.started_at,
            "expanded_budget": expanded_budget,
            "window_limits": window_limits,
        }
        artifacts["escalation_request"] = self._write_json(mission_dir / "escalation_request.json", request_payload)
        decision_payload = {
            "granted": True,
            "granted_at": escalation_state.started_at,
            "expanded_budget": expanded_budget,
        }
        artifacts["escalation_decision"] = self._write_json(mission_dir / "escalation_decision.json", decision_payload)
        window_payload = {
            "started_at": escalation_state.started_at,
            "limits": window_limits,
            "expanded_budget": expanded_budget,
        }
        artifacts["escalation_window"] = self._write_json(mission_dir / "escalation_window.json", window_payload)
        summary_text = (
            "Escalation granted due to risk budget exceedance."
            " Expanded steps/time/risk allowed within single bounded window."
        )
        (mission_dir / "escalation_summary.txt").write_text(summary_text, encoding="utf-8")
        artifacts["escalation_summary"] = str(mission_dir / "escalation_summary.txt")
        if trace_recorder:
            trace_recorder.record_lifecycle_event(
                event="escalation_requested",
                data={"reason": "risk_budget_exceeded", "detail": limit_detail},
            )
            trace_recorder.record_lifecycle_event(
                event="escalation_granted",
                data={"expanded_budget": expanded_budget, "window_limits": window_limits},
            )
        return artifacts

    def _check_escalation_window(
        self,
        *,
        mission_dir: Path,
        trace_recorder: ExecutionTraceRecorder | None,
        escalation_state: EscalationState,
        budget_monitor: BudgetMonitor,
        base_budget: AutonomyBudget,
        artifacts: Dict[str, str],
    ) -> None:
        if not escalation_state.used or not escalation_state.started_at:
            return
        if escalation_state.ended_at:
            return
        try:
            started_at = datetime.fromisoformat(escalation_state.started_at)
        except Exception:
            started_at = datetime.now(UTC)
        time_limit = float(escalation_state.window_limits.get("time_limit_sec", ESCALATION_TIME_BONUS_SEC))
        time_exceeded = (datetime.now(UTC) - started_at).total_seconds() > time_limit
        step_limit = budget_monitor.budget.max_steps
        time_to_close = time_exceeded or budget_monitor.usage.steps_used >= step_limit
        if time_to_close:
            escalation_state.mark_closed("window_closed")
            budget_monitor.budget = base_budget
            window_payload = {
                "started_at": escalation_state.started_at,
                "ended_at": escalation_state.ended_at,
                "limits": escalation_state.window_limits,
                "expanded_budget": escalation_state.expanded_budget,
                "closed_reason": "time_or_step_limit",
            }
            artifacts["escalation_window"] = self._write_json(mission_dir / "escalation_window.json", window_payload)
            summary_text = (
                "Escalation window closed after bounded exploration."
                " Execution continues with base autonomy budget."
            )
            (mission_dir / "escalation_summary.txt").write_text(summary_text, encoding="utf-8")
            if trace_recorder:
                trace_recorder.record_lifecycle_event(
                    event="escalation_closed",
                    data={"ended_at": escalation_state.ended_at, "reason": "time_or_step_limit"},
                )

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)

    async def _run_demo_force_actions(
        self,
        *,
        mission_spec: MissionSpec,
        mission_dir: Path,
        trace_recorder: ExecutionTraceRecorder | None,
    ) -> None:
        """Run a minimal visible bootstrap: navigate + search + scroll.

        This bypasses planner/judgment for first actions to prove the agent is live.
        Best-effort only; failures should not crash the mission.
        """

        try:
            worker = self._build_worker(mission_spec)
            run_logger = MissionArtifactLogger(base_dir=mission_dir / "demo_bootstrap", goal_name="demo_force_actions")
            worker.logger = run_logger
            worker.set_mission_context(mission_instruction=mission_spec.instruction, subgoal_description="demo_force_actions")
            actions = [
                {"action": "navigate", "url": "https://duckduckgo.com/?q=AI+startup+controversy"},
                {"action": "scroll", "direction": "down", "amount": 2000},
            ]
            result = await worker.execute({"action": actions, "goal": "demo_force_actions", "demo": True})
            artifacts = run_logger.to_dict()
            if trace_recorder:
                trace_recorder.record_artifact("demo_force_actions", str(mission_dir / "demo_bootstrap"))
                trace_recorder.record_lifecycle_event(
                    event="demo_force_actions",
                    data={
                        "status": result.get("error") or "ok",
                        "artifacts": artifacts,
                    },
                )
        except Exception:
            self.logger.warning("demo_force_actions failed; continuing", exc_info=True)

    async def _run_goal_driven_autonomous_loop(self, mission_spec: MissionSpec) -> MissionResult:
        start_ts = datetime.now(UTC)
        mission_dir = self._build_mission_dir(mission_spec, start_ts)
        goals = self._parse_goal_spec(mission_spec.instruction)
        required_fields = self._required_fields_for_mission(mission_spec.instruction)
        execution_context: Dict[str, Any] = {
            "visited_domains": set(),
            "visited_url_set": set(),
            "visited_urls": [],
            "open_tabs": [],
            "findings": {
                "items": [],
                "entities": goals["entities"],
                "actions": goals["actions"],
                "required_fields": required_fields,
                "schema": {field: [] if field in {"founders", "recent_mentions"} else "" for field in required_fields},
            },
            "extracted_entities": {
                "companies": set(),
                "people": set(),
                "websites": set(),
                "recent_mentions": [],
            },
            "founder_occurrences": {},
            "founder_domain_occurrences": {},
            "blocked_domains": set(),
            "field_provenance": {},
            "domain_link_stats": {},
            "navigation_links": 0,
            "total_links": 0,
            "steps_used": 0,
            "max_steps": 35,
            "sources_used": [],
            "tabs_opened": 0,
        }

        print("[GOAL] Planning")
        browser: Browser | None = None
        context: BrowserContext | None = None
        playwright = None
        goal_satisfied = False
        stop_reason = "step_budget_reached"
        no_high_score_rounds = 0
        hard_safety_halt = False

        try:
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(
                headless=False,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
            )
            context = await browser.new_context()

            entry_url = self._extract_url_from_text(mission_spec.instruction)
            if entry_url:
                primary_tab = await context.new_page()
                execution_context["open_tabs"].append(primary_tab)
                execution_context["tabs_opened"] += 1
                await primary_tab.goto(entry_url, wait_until="domcontentloaded")
            else:
                print("[GOAL] Searching")
                primary_tab = await context.new_page()
                execution_context["open_tabs"].append(primary_tab)
                execution_context["tabs_opened"] += 1
                await primary_tab.goto("https://www.google.com", wait_until="domcontentloaded")
                await primary_tab.fill("textarea[name='q']", mission_spec.instruction)
                await primary_tab.keyboard.press("Enter")
                await primary_tab.wait_for_timeout(1200)
                opened = await self._open_scored_links_from_page(primary_tab, context, execution_context, goals, top_n=2)
                if opened == 0:
                    await primary_tab.goto(f"https://duckduckgo.com/?q={quote_plus(mission_spec.instruction)}", wait_until="domcontentloaded")
                    await primary_tab.wait_for_timeout(1000)
                    opened = await self._open_scored_links_from_page(primary_tab, context, execution_context, goals, top_n=2)
                if opened == 0:
                    no_high_score_rounds += 1

            current_idx = 0
            while execution_context["steps_used"] < execution_context["max_steps"]:
                open_tabs: List[Page] = [tab for tab in execution_context["open_tabs"] if not tab.is_closed()]
                execution_context["open_tabs"] = open_tabs
                if not open_tabs:
                    stop_reason = "no_open_tabs"
                    break

                current_idx = min(current_idx, len(open_tabs) - 1)
                page = open_tabs[current_idx]
                domain = self._source_host(page.url) or "unknown"
                print(f"[GOAL] Switching tab: {domain}")
                await page.bring_to_front()
                await page.wait_for_timeout(300)

                page_url = page.url or ""
                if page_url and page_url not in execution_context["visited_url_set"]:
                    execution_context["visited_url_set"].add(page_url)
                    execution_context["visited_urls"].append(page_url)
                host = self._source_host(page_url)
                if host:
                    execution_context["visited_domains"].add(host)
                if host and host not in {"google.com", "www.google.com", "duckduckgo.com", "www.duckduckgo.com"}:
                    if page_url and page_url not in execution_context["sources_used"]:
                        execution_context["sources_used"].append(page_url)

                if self._is_hard_safety_url(page_url):
                    hard_safety_halt = True
                    stop_reason = "hard_safety_halt"
                    break

                if await self._is_blocked_challenge_page(page):
                    if host:
                        execution_context["blocked_domains"].add(host)
                    print("[GOAL] Source blocked — pivoting to alternative")
                    current_idx = (current_idx + 1) % len(execution_context["open_tabs"])
                    execution_context["steps_used"] += 1
                    continue

                page_type = await self._classify_current_page(page)
                opened_this_round = 0

                if page_type == "search_results":
                    opened_this_round += await self._open_scored_links_from_page(page, context, execution_context, goals, top_n=2)
                    if opened_this_round > 0:
                        current_idx = len(execution_context["open_tabs"]) - 1
                elif page_type == "content_page":
                    extracted = await self._extract_structured_data(page)
                    if extracted:
                        extracted["url"] = page_url
                        self._record_page_link_signal(execution_context, host or "unknown", extracted)
                        execution_context["findings"]["items"].append(extracted)
                        source_score = rank_source(host or "")
                        field_logs = self._merge_extracted_fields(
                            execution_context,
                            extracted,
                            host or "unknown",
                            source_score,
                        )
                        for field_name in field_logs:
                            print(f"[GOAL] Extracted {field_name} from {host or 'unknown'}")
                    if self._goals_satisfied(goals, execution_context):
                        print("[GOAL] Goal satisfied")
                        goal_satisfied = True
                        stop_reason = "goal_satisfied"
                        break

                    missing_fields = self._missing_required_fields(execution_context)
                    if missing_fields:
                        search_tabs = [
                            tab
                            for tab in execution_context["open_tabs"]
                            if not tab.is_closed()
                            and self._source_host(tab.url) in {"google.com", "www.google.com", "duckduckgo.com", "www.duckduckgo.com"}
                        ]
                        if search_tabs:
                            followup_query = self._build_followup_query(goals, missing_fields)
                            print("[GOAL] Searching")
                            await self._run_search_query(search_tabs[0], followup_query)
                            opened_this_round += await self._open_scored_links_from_page(search_tabs[0], context, execution_context, goals, top_n=2)

                    opened_this_round = await self._open_scored_links_from_page(page, context, execution_context, goals, top_n=2)
                    if opened_this_round > 0:
                        current_idx = len(execution_context["open_tabs"]) - 1
                elif page_type == "form":
                    needed = any(token in mission_spec.instruction.lower() for token in {"login", "sign in", "register", "form"})
                    if needed:
                        await self._fill_minimal_required_fields(page)
                    else:
                        await page.go_back(wait_until="domcontentloaded")
                else:
                    current_idx = (current_idx + 1) % len(execution_context["open_tabs"])

                no_high_score_rounds = 0 if opened_this_round > 0 else no_high_score_rounds + 1
                if no_high_score_rounds >= 6:
                    stop_reason = "no_high_scoring_links"
                    break

                execution_context["steps_used"] += 1
                await self._prune_low_value_tabs(execution_context, current_idx)
                if execution_context["open_tabs"]:
                    current_idx = (current_idx + 1) % len(execution_context["open_tabs"])

            if hard_safety_halt:
                goal_satisfied = False

            source_analysis = self._build_source_analysis(execution_context)
            low_signal_domains = self._build_low_signal_domains(execution_context, source_analysis)
            structured_result = {
                **execution_context["findings"],
                "schema": self._json_safe_schema(execution_context["findings"].get("schema", {})),
                "founder_occurrences": execution_context.get("founder_occurrences", {}),
            }
            executive_summary = build_executive_summary(structured_result, source_analysis)
            confidence_score = float(executive_summary.get("confidence_score", 0.0))

            founder_consensus_count = self._compute_founder_consensus_count(execution_context)
            consensus_boost = min(0.20, founder_consensus_count * 0.08)
            confidence_score += consensus_boost
            print(f"[GOAL] Cross-source founder consensus: {founder_consensus_count} matched")

            low_signal_penalty = self._compute_low_signal_penalty(execution_context, low_signal_domains)
            if low_signal_penalty > 0:
                confidence_score -= low_signal_penalty
                print(f"[GOAL] Low signal penalty applied: -{low_signal_penalty:.2f}")

            diversity_bonus = self._compute_source_diversity_bonus(source_analysis, low_signal_domains)
            if diversity_bonus > 0:
                confidence_score += diversity_bonus

            has_all_required_fields = not self._missing_required_fields(execution_context)
            high_signal_domains_count = self._count_high_signal_domains(source_analysis, low_signal_domains)
            if has_all_required_fields and high_signal_domains_count >= 2 and founder_consensus_count >= 2:
                confidence_score = max(confidence_score, 0.80)

            confidence_score = round(max(0.0, min(1.0, confidence_score)), 2)
            executive_summary["confidence_score"] = confidence_score

            if not executive_summary:
                stop_reason = "missing_executive_summary"

            signal_metrics = {
                "unique_domains": int(source_analysis.get("domains_count", 0) or 0),
                "founder_consensus_count": founder_consensus_count,
                "low_signal_penalty": round(low_signal_penalty, 2),
                "diversity_bonus": round(diversity_bonus, 2),
            }

            final_report = {
                "mission": mission_spec.instruction,
                "sources_used": execution_context["sources_used"],
                "source_analysis": source_analysis,
                "tabs_opened": execution_context["tabs_opened"],
                "steps_used": execution_context["steps_used"],
                "visited_urls": execution_context["visited_urls"],
                "visited_domains": sorted(execution_context["visited_domains"]),
                "structured_result": structured_result,
                "extracted_entities": self._json_safe_entities(execution_context["extracted_entities"]),
                "executive_summary": executive_summary,
                "confidence_score": confidence_score,
                "signal_metrics": signal_metrics,
                "goal_completion_reason": stop_reason,
                "output_format": goals["output_format"],
            }
            report_path = mission_dir / "final_report.json"
            report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
            print("[GOAL] Executive summary generated")
            print(f"[GOAL] Confidence score: {confidence_score:.2f}")

            end_ts = datetime.now(UTC)
            completion_ok = bool(goal_satisfied and executive_summary)
            completion_payload = {
                "complete": completion_ok,
                "reason": stop_reason,
                "payload": {"report": str(report_path)},
            }
            subgoal = MissionSubgoalResult(
                subgoal_id=f"{mission_spec.id}_goal_loop",
                description="Goal-driven autonomous loop",
                status="complete" if completion_ok else "failed",
                attempts=1,
                started_at=start_ts,
                ended_at=end_ts,
                completion=completion_payload,
                error=None if completion_ok else stop_reason,
                artifacts={"final_report": str(report_path)},
            )
            return MissionResult(
                mission_id=mission_spec.id,
                status="complete" if completion_ok else "failed",
                start_ts=start_ts,
                end_ts=end_ts,
                subgoal_results=[subgoal],
                summary={
                    "reason": stop_reason,
                    "goal_loop": True,
                    "sources_used": len(execution_context["sources_used"]),
                    "tabs_opened": execution_context["tabs_opened"],
                    "steps_used": execution_context["steps_used"],
                    "required_fields": required_fields,
                },
                artifacts_path=str(mission_dir),
                termination={"hard_safety_halt": hard_safety_halt} if hard_safety_halt else {},
            )
        except Exception as exc:  # noqa: BLE001
            end_ts = datetime.now(UTC)
            subgoal = MissionSubgoalResult(
                subgoal_id=f"{mission_spec.id}_goal_loop",
                description="Goal-driven autonomous loop",
                status="failed",
                attempts=1,
                started_at=start_ts,
                ended_at=end_ts,
                completion={"complete": False, "reason": "goal_loop_error", "payload": {"error": str(exc)}},
                error=str(exc),
                artifacts={},
            )
            return MissionResult(
                mission_id=mission_spec.id,
                status="failed",
                start_ts=start_ts,
                end_ts=end_ts,
                subgoal_results=[subgoal],
                summary={"reason": "goal_loop_error", "error": str(exc), "goal_loop": True},
                artifacts_path=str(mission_dir),
                termination={},
            )
        finally:
            if context is not None:
                await context.close()
            if browser is not None:
                await browser.close()
            if playwright is not None:
                await playwright.stop()

    def _parse_goal_spec(self, mission_text: str) -> Dict[str, Any]:
        stopwords = {
            "find", "open", "search", "extract", "return", "recent", "multiple", "sources", "structured", "and", "the", "for", "with", "from", "research",
        }
        entities = [
            token
            for token in re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", mission_text)
            if token.lower() not in stopwords
        ]
        verbs = re.findall(r"\b(open|search|extract|summarize|compare|analyze|find|collect|visit|research)\b", mission_text.lower())
        return {
            "entities": list(dict.fromkeys(entities))[:16],
            "actions": list(dict.fromkeys(verbs)) or ["search", "extract"],
            "output_format": "structured_json",
        }

    def _required_fields_for_mission(self, mission_text: str) -> List[str]:
        text = mission_text.lower()
        fields: List[str] = []
        if any(token in text for token in {"founder", "cofounder", "team", "ceo"}):
            fields.append("founders")
        if any(token in text for token in {"description", "profile", "business model", "what it does"}):
            fields.append("description")
        if any(token in text for token in {"website", "site", "homepage", "url"}):
            fields.append("website")
        if any(token in text for token in {"news", "mentions", "coverage", "recent"}):
            fields.append("recent_mentions")
        if not fields:
            fields = ["description", "recent_mentions"]
        return fields

    def _extract_url_from_text(self, mission_text: str) -> str | None:
        match = re.search(r"https?://[^\s\)]+", mission_text)
        if not match:
            return None
        return match.group(0).rstrip(".,)")

    async def _classify_current_page(self, page: Page) -> str:
        url = (page.url or "").lower()
        if url.startswith("about:blank"):
            return "unknown"
        if "google.com/search" in url or ("duckduckgo.com" in url and "q=" in url):
            return "search_results"
        password_inputs = await page.locator("input[type='password']").count()
        required_inputs = await page.locator("input[required], textarea[required], select[required]").count()
        if password_inputs > 0 or required_inputs >= 3:
            return "form"
        title = (await page.title()).strip().lower()
        if "search" in title and "result" in title:
            return "search_results"
        return "content_page"

    async def _open_scored_links_from_page(
        self,
        page: Page,
        context: BrowserContext,
        execution_context: Dict[str, Any],
        goals: Dict[str, Any],
        top_n: int = 2,
    ) -> int:
        rows: Any = []
        for _ in range(2):
            try:
                rows = await page.evaluate(
                    """
                    () => {
                      const anchors = Array.from(document.querySelectorAll('a[href]'));
                      const items = [];
                      for (const a of anchors) {
                        const href = a.href || '';
                        const titleNode = a.querySelector('h3');
                        const title = (titleNode ? titleNode.textContent : a.textContent || '').trim();
                        if (!href || !title) continue;
                        items.push({ title, url: href, text: (a.textContent || '').trim() });
                        if (items.length >= 80) break;
                      }
                      return items;
                    }
                    """
                )
                break
            except Exception:
                await page.wait_for_timeout(250)
                continue
        if not isinstance(rows, list) or not rows:
            return 0
        keywords = set(token.lower() for token in goals.get("entities", []) + goals.get("actions", []))
        seen_hosts = {
            self._source_host(url)
            for url in (list(execution_context.get("visited_urls", [])) + list(execution_context.get("sources_used", [])))
            if self._source_host(url)
        }
        blocked_domains = set(execution_context.get("blocked_domains", set()))
        seen_entities = {
            *{str(v).lower() for v in execution_context["extracted_entities"].get("companies", set())},
            *{str(v).lower() for v in execution_context["extracted_entities"].get("people", set())},
        }
        scored: List[Tuple[float, Dict[str, str]]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_url = str(row.get("url") or "").strip()
            title = str(row.get("title") or "").strip()
            text = str(row.get("text") or "").strip()
            url = self._normalize_result_url(raw_url)
            host = self._source_host(url)
            if not url or not host:
                continue
            if host in blocked_domains:
                continue
            blocked_hosts = {
                "google.com",
                "www.google.com",
                "support.google.com",
                "duckduckgo.com",
                "www.duckduckgo.com",
                "bing.com",
                "www.bing.com",
                "search.yahoo.com",
            }
            if host in blocked_hosts or host.endswith(".google.com"):
                continue
            if url in execution_context["visited_url_set"]:
                continue
            relevance = float(sum(1 for kw in keywords if kw and kw in f"{title} {text} {url}".lower()))
            diversity_bonus = 2.0 if host not in seen_hosts else 0.0
            entity_tokens = set(token.lower() for token in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", f"{title} {text}"))
            novelty_bonus = 1.5 if entity_tokens and not (entity_tokens & seen_entities) else 0.0
            source_score = rank_source(host)
            low_signal_penalty = 1.25 if self._is_base_low_signal_domain(host) else 0.0
            score = relevance + diversity_bonus + novelty_bonus + (source_score * 1.5) - low_signal_penalty
            if score <= 0:
                continue
            scored.append((score, {"url": url, "title": title, "source_score": source_score, "domain": host}))

        scored.sort(key=lambda row: row[0], reverse=True)
        opened = 0
        opened_hosts = {self._source_host(tab.url) for tab in execution_context["open_tabs"] if not tab.is_closed()}
        for _, candidate in scored:
            if opened >= top_n:
                break
            url = candidate["url"]
            host = self._source_host(url)
            if not host or host in opened_hosts:
                continue
            if len(execution_context["open_tabs"]) >= 7:
                break
            print(f"[GOAL] Ranked source: {candidate.get('domain', host)} ({float(candidate.get('source_score', 0.0)):.2f})")
            print(f"[GOAL] Opening new tab: {url}")
            new_tab = await context.new_page()
            try:
                await new_tab.goto(url, wait_until="domcontentloaded")
            except Exception:
                await new_tab.close()
                continue
            execution_context["open_tabs"].append(new_tab)
            execution_context["tabs_opened"] += 1
            opened += 1
            opened_hosts.add(host)
        return opened

    def _normalize_result_url(self, url: str) -> str:
        value = url.strip()
        if not value:
            return ""
        if "google.com/url?" in value:
            parsed = urlparse(value)
            extracted = parse_qs(parsed.query).get("q", [""])[0]
            if extracted:
                value = unquote(extracted)
        if "duckduckgo.com/l/?" in value:
            parsed = urlparse(value)
            extracted = parse_qs(parsed.query).get("uddg", [""])[0]
            if extracted:
                value = unquote(extracted)
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"}:
            return ""
        return value

    async def _extract_structured_data(self, page: Page) -> Dict[str, Any]:
        return await page.evaluate(
            """
            () => {
              const title = (document.querySelector('h1')?.textContent || document.title || '').trim();
              const description = (document.querySelector('meta[name="description"]')?.content || '').trim();
              const headings = Array.from(document.querySelectorAll('h1, h2, h3')).map(h => (h.textContent || '').trim()).filter(Boolean).slice(0, 10);
              const snippets = Array.from(document.querySelectorAll('p')).map(p => (p.textContent || '').trim()).filter(t => t.length > 50).slice(0, 8);
                            const allAnchors = Array.from(document.querySelectorAll('a[href]'));
                            const links = allAnchors.map(a => ({
                href: a.href,
                text: (a.textContent || '').trim(),
              })).filter(x => x.href && x.href.startsWith('http')).slice(0, 120);
              const founderText = Array.from(document.querySelectorAll('p, li, span, h2, h3')).map(n => (n.textContent || '').trim()).filter(Boolean).slice(0, 200);
              return {
                url: window.location.href,
                title,
                description,
                headings,
                snippets,
                links,
                                total_links: allAnchors.length,
                founder_text: founderText,
              };
            }
            """
        )

    def _merge_extracted_fields(
        self,
        execution_context: Dict[str, Any],
        extracted: Dict[str, Any],
        domain: str,
        source_score: float,
    ) -> List[str]:
        schema = execution_context["findings"].setdefault("schema", {})
        provenance: Dict[str, Dict[str, Any]] = execution_context.setdefault("field_provenance", {})
        required_fields: List[str] = execution_context["findings"].get("required_fields", [])
        logged: List[str] = []
        title = str(extracted.get("title") or "").strip()
        description = clean_description(str(extracted.get("description") or "").strip())
        snippets = [str(item) for item in (extracted.get("snippets") or []) if str(item).strip()]
        links = extracted.get("links") or []
        founder_text = [str(item) for item in (extracted.get("founder_text") or [])]

        if "description" in required_fields and not schema.get("description"):
            value = description or (snippets[0] if snippets else "")
            if value:
                if self._should_replace_field(provenance.get("description"), source_score):
                    schema["description"] = value
                    provenance["description"] = {"domain": domain, "score": source_score}
                    logged.append("description")

        if "website" in required_fields:
            candidates = [str(item.get("href") or "") for item in links if isinstance(item, dict)]
            preferred = next(iter(execution_context["findings"].get("entities", []) or []), "")
            external = validate_website(candidates, preferred_domain_token=preferred)
            if external:
                if self._should_replace_field(provenance.get("website"), source_score):
                    schema["website"] = external
                    provenance["website"] = {"domain": domain, "score": source_score}
                    execution_context["extracted_entities"].setdefault("websites", set()).add(external)
                    logged.append("website")

        if "founders" in required_fields:
            founders = schema.get("founders") or []
            founder_candidates: List[str] = []
            for line in founder_text[:120]:
                compact_line = " ".join((line or "").split())
                if not re.search(r"\b(founder|co-?founder|ceo|founding)\b", compact_line, flags=re.IGNORECASE):
                    continue
                if len(compact_line) > 220:
                    continue
                if len(re.findall(r"\b[A-Z][a-z'\-]+\b", compact_line)) > 8:
                    continue
                for name in re.findall(r"\b[A-Z][a-z'\-]+\s+[A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+)?\b", compact_line):
                    founder_candidates.append(name)
            validated = validate_founders(founder_candidates, execution_context.get("founder_occurrences"))
            for name in validated:
                if name not in founders:
                    founders.append(name)
                    print(f"[GOAL] Extracted validated founder: {name}")
                normalized = self._normalize_founder_name(name)
                if normalized:
                    by_domain = execution_context.setdefault("founder_domain_occurrences", {}).setdefault(normalized, set())
                    by_domain.add(domain)
            if founders and self._should_replace_field(provenance.get("founders"), source_score):
                schema["founders"] = founders[:8]
                provenance["founders"] = {"domain": domain, "score": source_score}
                execution_context["extracted_entities"].setdefault("people", set()).update(founders)
                if any(int(count) >= 2 for count in execution_context.get("founder_occurrences", {}).values()):
                    print("[GOAL] Cross-source validation passed")
                logged.append("founders")

        if "recent_mentions" in required_fields:
            mentions = schema.get("recent_mentions") or []
            mention_candidates: List[Dict[str, str]] = []
            for item in links:
                if not isinstance(item, dict):
                    continue
                href = str(item.get("href") or "")
                text = str(item.get("text") or "")
                if not href:
                    continue
                if re.search(r"news|blog|press|article|coverage|post", f"{href} {text}", flags=re.IGNORECASE):
                    mention_candidates.append({"title": text or href, "url": href})
            mentions.extend(mention_candidates)
            mentions = dedupe_mentions(mentions)
            if mentions:
                if self._should_replace_field(provenance.get("recent_mentions"), source_score):
                    schema["recent_mentions"] = mentions[:10]
                    provenance["recent_mentions"] = {"domain": domain, "score": source_score}
                execution_context["extracted_entities"].setdefault("recent_mentions", []).extend(mentions[:10])
                logged.append("recent_mentions")

        if title:
            execution_context["extracted_entities"].setdefault("companies", set()).add(title)

        return list(dict.fromkeys(logged))

    def _should_replace_field(self, existing: Dict[str, Any] | None, candidate_score: float) -> bool:
        if not existing:
            return True
        current = float(existing.get("score", 0.0))
        return candidate_score >= current

    async def _discover_relevant_links(
        self,
        page: Page,
        goals: Dict[str, Any],
        execution_context: Dict[str, Any],
    ) -> List[str]:
        links = await page.evaluate(
            """
            () => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                href: a.href,
                text: (a.textContent || '').trim(),
            })).filter(x => !!x.href)
            """
        )
        if not isinstance(links, list):
            return []
        keywords = set(token.lower() for token in goals.get("entities", []) + goals.get("actions", []))
        seen_domains = set(execution_context.get("visited_domains", set()))
        seen_entities = {
            *{str(v).lower() for v in execution_context["extracted_entities"].get("companies", set())},
            *{str(v).lower() for v in execution_context["extracted_entities"].get("people", set())},
        }
        ranked: List[Tuple[float, str]] = []
        for item in links:
            if not isinstance(item, dict):
                continue
            href = self._normalize_result_url(str(item.get("href") or ""))
            if not href or href in execution_context["visited_url_set"]:
                continue
            if href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            blob_raw = f"{href} {item.get('text', '')}"
            blob = blob_raw.lower()
            score = float(sum(1 for kw in keywords if kw and kw in blob))
            domain = self._source_host(href)
            if domain and domain not in seen_domains:
                score += 2.0
            entity_tokens = set(token.lower() for token in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", blob_raw))
            if entity_tokens and not (entity_tokens & seen_entities):
                score += 1.0
            if score <= 0:
                continue
            ranked.append((score, href))
        ranked.sort(key=lambda row: row[0], reverse=True)
        return [href for _, href in ranked[:5]]

    async def _prune_low_value_tabs(self, execution_context: Dict[str, Any], current_idx: int) -> None:
        open_tabs: List[Page] = [tab for tab in execution_context["open_tabs"] if not tab.is_closed()]
        while len(open_tabs) > 7:
            victim_idx = 0 if current_idx != 0 else 1
            victim = open_tabs[victim_idx]
            await victim.close()
            del open_tabs[victim_idx]
            if current_idx >= len(open_tabs):
                current_idx = max(0, len(open_tabs) - 1)
        execution_context["open_tabs"] = open_tabs

    def _goals_satisfied(self, goals: Dict[str, Any], execution_context: Dict[str, Any]) -> bool:
        required_fields: List[str] = execution_context["findings"].get("required_fields", [])
        schema: Dict[str, Any] = execution_context["findings"].get("schema", {})
        source_domains = {
            self._source_host(str(url))
            for url in execution_context.get("sources_used", [])
            if self._source_host(str(url))
        }
        if len(source_domains) < 3:
            return False
        high_signal_domains = {domain for domain in source_domains if not self._is_base_low_signal_domain(domain)}
        if len(high_signal_domains) < 2:
            return False
        if len(execution_context.get("visited_urls", [])) < 3:
            return False
        for field in required_fields:
            value = schema.get(field)
            if isinstance(value, list) and not value:
                return False
            if isinstance(value, str) and not value.strip():
                return False
            if value is None:
                return False
        findings = execution_context["findings"].get("items", [])
        if not findings:
            return False
        query_tokens = [token.lower() for token in goals.get("entities", []) if token]
        if query_tokens:
            corpus = json.dumps(findings).lower()
            if not any(token in corpus for token in query_tokens):
                return False
        return True

    async def _fill_minimal_required_fields(self, page: Page) -> None:
        required_fields = page.locator("input[required], textarea[required]")
        count = await required_fields.count()
        for idx in range(min(count, 3)):
            field = required_fields.nth(idx)
            field_type = ((await field.get_attribute("type")) or "text").lower()
            try:
                if field_type == "email":
                    await field.fill("research@example.com")
                else:
                    await field.fill("research")
            except Exception:
                continue

    def _is_hard_safety_url(self, url: str) -> bool:
        lowered = (url or "").lower()
        return lowered.startswith("file://") or lowered.startswith("chrome://")

    def _missing_required_fields(self, execution_context: Dict[str, Any]) -> List[str]:
        required_fields: List[str] = execution_context["findings"].get("required_fields", [])
        schema: Dict[str, Any] = execution_context["findings"].get("schema", {})
        missing: List[str] = []
        for field in required_fields:
            value = schema.get(field)
            if isinstance(value, list):
                if not value:
                    missing.append(field)
            elif isinstance(value, str):
                if not value.strip():
                    missing.append(field)
            elif value is None:
                missing.append(field)
        return missing

    def _build_followup_query(self, goals: Dict[str, Any], missing_fields: List[str]) -> str:
        subject = " ".join(goals.get("entities", [])[:4]).strip() or "company"
        fields = " ".join(missing_fields)
        return f"{subject} {fields}".strip()

    async def _is_blocked_challenge_page(self, page: Page) -> bool:
        try:
            title = (await page.title() or "").lower()
        except Exception:
            title = ""
        body_text = ""
        try:
            body = page.locator("body")
            if await body.count() > 0:
                body_text = ((await body.first.inner_text(timeout=1500)) or "").lower()
        except Exception:
            body_text = ""
        corpus = f"{title}\n{body_text[:4000]}"
        markers = [
            "access denied",
            "verify you are human",
            "are you a robot",
            "unusual traffic",
            "security check",
            "challenge",
            "captcha",
            "cloudflare",
            "temporarily blocked",
            "blocked",
        ]
        return any(marker in corpus for marker in markers)

    def _build_source_analysis(self, execution_context: Dict[str, Any]) -> Dict[str, Any]:
        domain_to_urls: Dict[str, List[str]] = {}
        for url in execution_context.get("sources_used", []):
            host = self._source_host(str(url))
            if not host:
                continue
            domain_to_urls.setdefault(host, []).append(str(url))
        domains = sorted(domain_to_urls.keys())
        top_sources: List[Dict[str, Any]] = []
        for domain in domains:
            score = float(rank_source(domain))
            top_sources.append(
                {
                    "domain": domain,
                    "score": score,
                    "bucket": bucket_source(domain),
                    "url_count": len(domain_to_urls.get(domain, [])),
                    "sample_url": (domain_to_urls.get(domain, [""])[0] if domain_to_urls.get(domain) else ""),
                }
            )
        top_sources.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        high = sum(1 for item in top_sources if str(item.get("bucket") or "").startswith("high"))
        medium = sum(1 for item in top_sources if str(item.get("bucket") or "").startswith("medium"))
        low = sum(1 for item in top_sources if str(item.get("bucket") or "").startswith("low"))
        avg = (sum(float(item.get("score", 0.0)) for item in top_sources) / len(top_sources)) if top_sources else 0.0
        return {
            "domains_count": len(domains),
            "high_credibility_count": high,
            "medium_credibility_count": medium,
            "low_credibility_count": low,
            "average_score": round(avg, 3),
            "top_sources": top_sources[:10],
        }

    def _normalize_founder_name(self, name: str) -> str:
        value = " ".join((name or "").strip().split())
        if not value:
            return ""
        value = re.sub(r"[\.,]", " ", value)
        tokens = [token for token in value.split() if token]
        cleaned: List[str] = []
        for token in tokens:
            lowered = token.lower()
            if lowered in _FOUNDER_TITLE_PREFIXES:
                continue
            cleaned.append(lowered)
        return " ".join(cleaned).strip()

    def _record_page_link_signal(self, execution_context: Dict[str, Any], domain: str, extracted: Dict[str, Any]) -> None:
        links = extracted.get("links") or []
        total_extracted = len(links) if isinstance(links, list) else 0
        nav_links = 0
        if isinstance(links, list):
            nav_patterns = re.compile(
                r"\b(home|about|contact|privacy|terms|cookie|login|sign in|signup|register|help|support|careers|sitemap|pricing|product|features)\b",
                flags=re.IGNORECASE,
            )
            nav_href_patterns = re.compile(
                r"/(about|contact|privacy|terms|cookie|careers|pricing|products?|features|customers?|offerings?|solutions?|news|blog|media|support|help|signup|register|login)(/|$)",
                flags=re.IGNORECASE,
            )
            for item in links:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip().lower()
                href = str(item.get("href") or "").strip().lower()
                href_host = self._source_host(href)
                word_count = len(text.split()) if text else 0
                short_label = 0 < word_count <= 4
                same_domain = bool(href_host and (href_host == domain or href_host.endswith(f".{domain}")))
                if not text:
                    nav_links += 1
                elif nav_patterns.search(text) or nav_patterns.search(href) or nav_href_patterns.search(href):
                    nav_links += 1
                elif short_label and same_domain:
                    nav_links += 1
                elif len(text) <= 2:
                    nav_links += 1
        execution_context["navigation_links"] = int(execution_context.get("navigation_links", 0) or 0) + nav_links
        execution_context["total_links"] = int(execution_context.get("total_links", 0) or 0) + total_extracted

        domain_stats = execution_context.setdefault("domain_link_stats", {}).setdefault(domain, {"max_total_links": 0})
        total_page_links = int(extracted.get("total_links") or 0)
        domain_stats["max_total_links"] = max(int(domain_stats.get("max_total_links", 0) or 0), total_page_links)

    def _build_low_signal_domains(self, execution_context: Dict[str, Any], source_analysis: Dict[str, Any]) -> set[str]:
        domains = {
            host
            for host in LOW_SIGNAL_BASE_DOMAINS
        }
        for source in source_analysis.get("top_sources", []):
            if not isinstance(source, dict):
                continue
            domain = str(source.get("domain") or "").lower()
            if self._is_base_low_signal_domain(domain):
                domains.add(domain)

        for domain, stats in execution_context.get("domain_link_stats", {}).items():
            if int((stats or {}).get("max_total_links", 0) or 0) > 150:
                domains.add(str(domain).lower())
        return domains

    def _compute_founder_consensus_count(self, execution_context: Dict[str, Any]) -> int:
        founder_domains = execution_context.get("founder_domain_occurrences", {})
        if not isinstance(founder_domains, dict):
            return 0
        return sum(1 for domains in founder_domains.values() if isinstance(domains, set) and len(domains) >= 2)

    def _compute_low_signal_penalty(self, execution_context: Dict[str, Any], low_signal_domains: set[str]) -> float:
        penalty = 0.0
        domain_counts: Counter[str] = Counter()
        for url in execution_context.get("sources_used", []):
            host = self._source_host(str(url))
            if host:
                domain_counts[host] += 1
        if domain_counts:
            dominant_domain, dominant_count = domain_counts.most_common(1)[0]
            total = sum(domain_counts.values())
            dominant_is_low_signal = (
                dominant_domain in low_signal_domains
                or self._is_base_low_signal_domain(dominant_domain)
            )
            if dominant_is_low_signal and total > 0 and (dominant_count / total) >= 0.5:
                penalty += 0.15

        total_links = int(execution_context.get("total_links", 0) or 0)
        navigation_links = int(execution_context.get("navigation_links", 0) or 0)
        if total_links > 0 and (navigation_links / total_links) > 0.5:
            penalty += 0.10
        return penalty

    def _count_high_signal_domains(self, source_analysis: Dict[str, Any], low_signal_domains: set[str]) -> int:
        count = 0
        for source in source_analysis.get("top_sources", []):
            if not isinstance(source, dict):
                continue
            domain = str(source.get("domain") or "").lower()
            if not domain:
                continue
            if domain in low_signal_domains:
                continue
            if self._is_base_low_signal_domain(domain):
                continue
            count += 1
        return count

    def _is_base_low_signal_domain(self, domain: str) -> bool:
        value = (domain or "").lower().strip()
        if not value:
            return False
        return any(value == base or value.endswith(f".{base}") for base in LOW_SIGNAL_BASE_DOMAINS)

    def _compute_source_diversity_bonus(self, source_analysis: Dict[str, Any], low_signal_domains: set[str]) -> float:
        unique_domains = int(source_analysis.get("domains_count", 0) or 0)
        high_signal_domains = self._count_high_signal_domains(source_analysis, low_signal_domains)
        if unique_domains >= 3 and high_signal_domains >= 2:
            return 0.12
        return 0.0

    async def _run_search_query(self, page: Page, query: str) -> None:
        url = (page.url or "").lower()
        if "google.com" in url:
            box = page.locator("textarea[name='q']")
            if await box.count() > 0:
                await box.first.fill(query)
                await page.keyboard.press("Enter")
                await page.wait_for_timeout(1000)
                return
        if "duckduckgo.com" in url:
            box = page.locator("input[name='q']")
            if await box.count() > 0:
                await box.first.fill(query)
                await page.keyboard.press("Enter")
                await page.wait_for_timeout(1000)
                return
        await page.goto(f"https://duckduckgo.com/?q={quote_plus(query)}", wait_until="domcontentloaded")
        await page.wait_for_timeout(1000)

    def _json_safe_entities(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        safe: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, set):
                safe[key] = sorted(value)
            elif isinstance(value, list):
                safe[key] = value
            else:
                safe[key] = value
        return safe

    def _json_safe_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        safe: Dict[str, Any] = {}
        for key, value in schema.items():
            if isinstance(value, set):
                safe[key] = sorted(value)
            else:
                safe[key] = value
        return safe

    def _source_host(self, url: str) -> str:
        try:
            return (urlparse(url).netloc or "").lower()
        except Exception:
            return ""


def run_mission_sync(
    mission_spec: MissionSpec,
    executor: MissionExecutor | None = None,
    *,
    resume_from: str | Path | None = None,
) -> MissionResult:
    """Convenience wrapper for synchronous callers (e.g., CLI)."""

    executor = executor or MissionExecutor()
    return asyncio.run(executor.run_mission(mission_spec, resume_from=resume_from))


def _slugify(value: str) -> str:
    cleaned = [ch if ch.isalnum() else "_" for ch in value.lower()]
    slug = "".join(cleaned).strip("_")
    return slug or "mission"


__all__ = ["MissionExecutor", "run_mission_sync"]


def _ensure_screenshot_action(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = [dict(action) for action in actions]
    for action in normalized:
        if (action.get("action") or "").lower() == "screenshot":
            return normalized
    normalized.append({"action": "screenshot", "name": "secure_area.png"})
    return normalized
