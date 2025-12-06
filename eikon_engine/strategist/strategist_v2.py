"""State-aware Strategist v2 implementation."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from typing import TYPE_CHECKING

from eikon_engine.core.completion import build_completion
from eikon_engine.core.strategist import StrategyStep
from eikon_engine.core.types import CompletionPayload
from eikon_engine.planning.memory_store import MemoryStore
from eikon_engine.planning.planner_v1 import PlannerV1, PlanState

from .agent_memory import AgentMemory, AgentMemoryHint
from .behavior_learner import BehaviorLearner
from .dom_features import DomFeatures, extract_dom_features, selector_in_dom
from .interference_detector import detect_interference
from .selector_healing import heal_selector
from .self_repair import SelfRepairEngine
from .state_detector import detect_state, find_cookie_popup
from eikon_engine.strategist_v2.navigator_reward_model import compute_reward
from eikon_engine.strategist_v2.confidence_scorer import score_decision
from eikon_engine.skills.skill_registry import SkillRegistry

if TYPE_CHECKING:
    from eikon_engine.stability import StabilityMonitor


PlanDict = Dict[str, Any]
StepMeta = Dict[str, Any]
RunContext = Dict[str, Any]
MicroRepair = Dict[str, Any]


@dataclass
class StrategistBase:
    """Base strategist capable of flattening planner tasks into steps."""

    planner: Any
    memory_store: MemoryStore | None = None
    run_trace: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.memory_store = self.memory_store or MemoryStore()
        self._plan: PlanDict | None = None
        self._steps: List[StepMeta] = []
        self._cursor = 0
        self._goal = ""
        self._last_result: Dict[str, Any] | None = None
        self._completion: CompletionPayload | None = None
        self._inserted_counter = itertools.count(1)
        self._goal_queue: List[Dict[str, Any]] = []
        self._last_dom_snapshot: str = ""
        self._reward_history: List[Dict[str, Any]] = []
        self._last_reward: float = 0.0
        self._confidence_state: Dict[str, Any] | None = None
        self._recovery_severity = 0
        self.self_repair = SelfRepairEngine()
        self.agent_memory = AgentMemory()
        self.adaptive_planner = PlannerV1()
        self.behavior_learner = BehaviorLearner()
        self._adaptive_plan: PlanState | None = None
        self._selector_bias: str = "css"
        self._anticipate_repair = False
        self._behavior_difficulty = 0.5
        self._stability_monitor: "StabilityMonitor | None" = None
        self._skill_registry = SkillRegistry

    async def initialize(self, goal: str) -> None:
        self._goal = goal
        self.run_trace.clear()
        self._completion = None
        self._last_result = None
        self._cursor = 0
        self._goal_queue.clear()
        self._last_dom_snapshot = ""
        self._reward_history.clear()
        self._last_reward = 0.0
        self._confidence_state = None
        self._recovery_severity = 0
        self._login_flow_triggered = False
        if self.self_repair:
            self.self_repair.reset()
        self._adaptive_plan = None
        await self._load_plan(goal)

    async def ensure_plan(self) -> None:
        if self._plan is None:
            await self._load_plan(self._goal)

    def has_next(self) -> bool:
        return self._cursor < len(self._steps)

    def attach_stability_monitor(self, monitor: "StabilityMonitor | None") -> None:
        self._stability_monitor = monitor

    def peek_step(self) -> StepMeta:
        if not self.has_next():
            raise StopIteration("Strategist has no pending steps")
        return self._steps[self._cursor]

    def next_step(self) -> StrategyStep:
        meta = self.peek_step()
        self._cursor += 1
        description = self._describe(meta)
        return StrategyStep(description=description, metadata=meta)

    def skip_current_step(self, *, reason: str) -> None:
        meta = self.peek_step()
        self._append_trace("skip", step_id=meta.get("step_id"), reason=reason)
        self._cursor += 1

    def insert_steps(self, actions: Sequence[Dict[str, Any]], *, bucket: str | None, tag: str) -> None:
        new_steps: List[StepMeta] = []
        bucket_name = bucket or "misc"
        task_id = f"{tag}_task"
        for action in actions:
            payload = dict(action)
            payload.setdefault("id", f"auto_{next(self._inserted_counter)}")
            step_meta: StepMeta = {
                "step_id": payload["id"],
                "task_id": payload.get("task_id", task_id),
                "bucket": bucket_name,
                "action": payload.get("action"),
                "selector": payload.get("selector"),
                "url": payload.get("url"),
                "action_payload": payload,
                "source": tag,
            }
            new_steps.append(step_meta)
        self._steps[self._cursor:self._cursor] = new_steps
        self._append_trace("insert", count=len(new_steps), tag=tag)

    def record_result(self, result: Dict[str, Any]) -> None:
        self._last_result = result
        completion = result.get("completion")
        error = result.get("error")
        is_completion = isinstance(completion, dict)
        if error or (is_completion and not completion.get("complete", True)):
            self._completion = completion if is_completion else build_completion(complete=False, reason=str(error or "worker error"), payload={})
            self._steps = []
            return
        if not self.has_next():
            if self._goal_queue:
                next_entry = self._goal_queue.pop(0)
                self._start_subgoal(next_entry)
            elif is_completion:
                self._completion = completion  # type: ignore[assignment]
            else:
                self._completion = build_completion(complete=True, reason="plan exhausted", payload={})

    def completion_state(self) -> CompletionPayload:
        if self._completion:
            return self._completion
        return build_completion(complete=False, reason="strategy pending", payload={})

    def load_plan(self, plan: PlanDict) -> None:
        self._plan = plan
        self._cursor = 0
        self._steps = self._flatten_plan(plan)

    async def _load_plan(self, goal: str) -> None:
        plan = await self._invoke_planner(goal, last_result=self._last_result)
        self.load_plan(plan)

    async def _invoke_planner(self, goal: str, *, last_result: Optional[Dict[str, Any]]) -> PlanDict:
        if last_result is None:
            return await self.planner.create_plan(goal)
        try:
            return await self.planner.create_plan(goal, last_result=last_result)
        except TypeError:
            return await self.planner.create_plan(goal)

    def _flatten_plan(self, plan: PlanDict) -> List[StepMeta]:
        steps: List[StepMeta] = []
        tasks = plan.get("tasks", [])
        for task in tasks:
            bucket = task.get("bucket", "misc")
            for action in task.get("inputs", {}).get("actions", []):
                payload = dict(action)
                payload.setdefault("id", f"step_{len(steps) + 1:03d}")
                step_meta: StepMeta = {
                    "step_id": payload["id"],
                    "task_id": task.get("id"),
                    "bucket": bucket,
                    "action": payload.get("action"),
                    "selector": payload.get("selector"),
                    "url": payload.get("url"),
                    "action_payload": payload,
                    "source": task.get("tool", "plan"),
                }
                steps.append(step_meta)
        return steps

    def _describe(self, meta: StepMeta) -> str:
        action = meta.get("action") or "step"
        target = meta.get("selector") or meta.get("url") or meta.get("bucket") or ""
        return f"{action}:{target}".strip(":")

    def _append_trace(self, event: str, **payload: Any) -> None:
        entry = {"event": event, **payload, "cursor": self._cursor}
        self.run_trace.append(entry)

    def queue_subgoal(self, goal: str, *, plan: PlanDict | None = None) -> None:
        self._goal_queue.append({"goal": goal, "plan": plan})
        self._append_trace("goal_chain_queue", goal=goal, pending=len(self._goal_queue))

    def _start_subgoal(self, entry: Dict[str, Any]) -> None:
        goal = entry.get("goal") or ""
        plan = entry.get("plan")
        self._goal = goal or self._goal
        self._completion = None
        if plan:
            self.load_plan(plan)
        else:
            self._plan = None
            self._steps = []
            self._cursor = 0
        self._append_trace("goal_chain_start", goal=self._goal, remaining=len(self._goal_queue))

    def finalize_run(
        self,
        run_ctx: RunContext,
        completion: CompletionPayload,
        duration_seconds: float,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._stability_monitor:
            return None
        artifact_base = None
        if artifacts:
            artifact_base = artifacts.get("base_dir")
        report = self._stability_monitor.evaluate_run(
            goal=self._goal,
            completion=completion,
            run_context=run_ctx,
            strategist_trace=self.run_trace,
            duration_seconds=duration_seconds,
            artifact_base=artifact_base,
        )
        fingerprint = run_ctx.get("current_fingerprint")
        if fingerprint:
            metrics = report.get("metrics") or {}
            self.agent_memory.store_stability(fingerprint, metrics)
        run_ctx["stability_summary"] = report
        return report


class StrategistV2(StrategistBase):
    """Strategist that adapts after every worker step using DOM signals."""

    def __init__(
        self,
        planner: Any,
        *,
        memory_store: MemoryStore | None = None,
        failure_budget: int = 3,
        failure_limit: int = 3,
    ) -> None:
        super().__init__(planner=planner, memory_store=memory_store)
        self.failure_budget = failure_budget
        self.failure_limit = failure_limit
        self._last_features: Optional[DomFeatures] = None
        self._last_mode: str = "unknown"
        self._repair_attempts = 0
        self._failure_counts: Dict[str, int] = {}
        self._last_url: str = ""
        self._recovery_counts: Dict[str, int] = {}
        self._login_flow_triggered = False

    def detect_state(self, dom: str, url: Optional[str] = None) -> Dict[str, Any]:
        state = detect_state(dom, url)
        self._last_features = state.get("features")
        self._last_mode = state.get("mode", "unknown")
        return state

    def on_step_result(self, run_ctx: RunContext, planned_step: StepMeta, step_outcome: Dict[str, Any]) -> None:
        dom = self._load_dom_snapshot(step_outcome)
        url = step_outcome.get("meta", {}).get("url") or planned_step.get("url") or run_ctx.get("current_url")
        run_ctx["active_subgoal"] = planned_step.get("task_id") or planned_step.get("bucket")
        state = self.detect_state(dom, url)
        run_ctx["current_url"] = url or run_ctx.get("current_url")
        if not run_ctx.get("force_replan"):
            run_ctx.pop("_behavior_replan_logged", None)
        run_ctx["current_fingerprint"] = self._page_fingerprint(run_ctx.get("current_url"), dom)
        run_ctx.setdefault("planner_events", [])
        payload = planned_step.get("action_payload", {})
        action = (payload.get("action") or "").lower()
        if action == "dom_presence_check" and state.get("mode") == "login_page":
            self._inject_login_flow(planned_step)
        intent = state.get("intent")
        if intent:
            run_ctx.setdefault("page_intents", []).append({"intent": getattr(intent, "intent", "unknown"), "confidence": getattr(intent, "confidence", 0.0)})
        run_ctx.setdefault("history", []).append({
            "step_id": planned_step.get("step_id"),
            "mode": state.get("mode"),
            "status": step_outcome.get("completion", {}).get("reason"),
        })
        artifact_meta = self._collect_failure_artifacts(step_outcome)
        if artifact_meta:
            run_ctx.setdefault("failure_artifacts", []).append({**artifact_meta, "step_id": planned_step.get("step_id")})
        if step_outcome.get("error"):
            self.record_failure(step_outcome.get("error", "error"))
            self._schedule_progressive_recovery(run_ctx, planned_step)
        if state.get("mode") == "dashboard_page":
            self._skip_login_steps()
        popup = find_cookie_popup(dom, state.get("features"))
        if popup:
            self._insert_cookie_dismiss(popup, planned_step)
        interference = detect_interference(dom, state.get("features"))
        if interference:
            self._insert_interference_actions(interference, planned_step)
        completion = step_outcome.get("completion") or {}
        step_failed = bool(step_outcome.get("error") or not completion.get("complete", True))
        selector = payload.get("selector")
        features = state.get("features")
        is_cookie_step = planned_step.get("source") == "cookie_popup"
        if selector and action in {"click", "fill", "dom_presence_check"}:
            features = features or extract_dom_features(dom)
            selector_missing = not selector_in_dom(selector, dom)
            navigated_away = bool(url and self._last_url and url != self._last_url)
            anticipatory = self._anticipate_repair and not navigated_away
            if navigated_away and selector_missing:
                selector_missing = False
            if selector_missing or anticipatory:
                repair = self.apply_micro_repair(planned_step, features)
                if repair.get("patched"):
                    if is_cookie_step:
                        self._apply_cookie_popup_patch(planned_step, repair)
                    else:
                        self._queue_repair_step(planned_step, repair)
        self._process_reward_signal(run_ctx, planned_step, dom, state)
        self._apply_behavior_prediction(run_ctx, run_ctx.get("current_fingerprint", ""), planned_step.get("step_id"))
        failure_reason = step_outcome.get("error") or completion.get("reason")
        failure_flag = bool(step_outcome.get("error") or (completion and not completion.get("complete", True)))
        recent_failure: Optional[str] = None
        if failure_flag:
            self.on_failure_detected({"step_id": planned_step.get("step_id"), "reason": failure_reason})
            self._attempt_self_repair(run_ctx, planned_step, dom, failure_reason)
            recent_failure = failure_reason
        elif self._confidence_state and self._confidence_state.get("band") == "low":
            self.on_failure_detected({"step_id": planned_step.get("step_id"), "reason": "low_confidence"})
            self._attempt_self_repair(run_ctx, planned_step, dom, "low_confidence")
            recent_failure = "low_confidence"
        self._apply_skill_suggestions(run_ctx, state, recent_failure, planned_step)
        if action == "navigate" and url and payload.get("url") and payload.get("url") != url:
            run_ctx.setdefault("redirects", []).append({"from": payload.get("url"), "to": url})
        self._last_url = url or self._last_url
        self._mark_active_subgoal(run_ctx, "failed" if failure_flag else "completed")
        self._update_adaptive_plan(run_ctx, state, dom, recent_failure if failure_flag else None)
        self._record_memory(run_ctx, dom)
        if completion.get("complete"):
            self._update_behavior_model(run_ctx, dom)
        elif run_ctx.get("force_replan") and not run_ctx.get("_behavior_replan_logged"):
            self._update_behavior_model(run_ctx, dom)
            run_ctx["_behavior_replan_logged"] = True

    def detect_state_only(self, dom: str, url: Optional[str] = None) -> str:
        return self.detect_state(dom, url).get("mode", "unknown")

    def apply_micro_repair(self, planned_step: StepMeta, features: DomFeatures) -> MicroRepair:
        payload = planned_step.get("action_payload", {})
        selector = payload.get("selector")
        action = (payload.get("action") or "").lower()
        if not selector:
            return {"patched": False, "reason": "no-selector"}
        bias = getattr(self, "_selector_bias", "css")
        candidates = self._find_similar_selector(selector, features)
        if candidates:
            self._repair_attempts += 1
            return {"patched": True, "new_selector": candidates, "reason": "loose_match"}
        label = payload.get("label") or payload.get("text") or self._infer_label_from_selector(selector)
        label_candidate = self._find_by_label(label, features) if label else None
        if bias == "text" and label_candidate:
            self._repair_attempts += 1
            return {"patched": True, "new_selector": label_candidate, "reason": "label_bias"}
        if label_candidate:
            self._repair_attempts += 1
            return {"patched": True, "new_selector": label_candidate, "reason": "label_match"}
        if action == "fill":
            inferred = self._infer_form_selector(selector, features)
            if inferred:
                self._repair_attempts += 1
                return {"patched": True, "new_selector": inferred, "reason": "form_inference"}
        healing_entry = self._heal_selector_via_module(planned_step, features)
        if bias == "role" and healing_entry:
            self._repair_attempts += 1
            return {
                "patched": True,
                "new_selector": healing_entry["selector"],
                "reason": f"selector_healing:{healing_entry['reason']}",
            }
        if healing_entry:
            self._repair_attempts += 1
            return {
                "patched": True,
                "new_selector": healing_entry["selector"],
                "reason": f"selector_healing:{healing_entry['reason']}",
            }
        return {"patched": False, "reason": "no-match"}

    def _process_reward_signal(self, run_ctx: RunContext, planned_step: StepMeta, dom: str, state: Dict[str, Any]) -> None:
        old_dom = self._last_dom_snapshot
        action_label = planned_step.get("action") or planned_step.get("action_payload", {}).get("action")
        reward_info = compute_reward(old_dom, dom, run_ctx.get("current_url"), action_label, run_ctx.get("active_subgoal"))
        self._last_dom_snapshot = dom
        self._last_reward = reward_info["reward"]
        payload = {**reward_info, "step_id": planned_step.get("step_id")}
        self._reward_history.append(payload)
        failure_count = self._max_failure_count()
        strategist_state = {"mode": state.get("mode"), "severity": self._recovery_severity}
        confidence_info = score_decision(self._last_reward, strategist_state, failure_count)
        self._confidence_state = confidence_info
        run_ctx.setdefault("reward_trace", []).append({
            "step_id": planned_step.get("step_id"),
            "reward": self._last_reward,
            "reasons": reward_info["reasons"],
            "confidence": confidence_info,
        })
        self._update_recovery_severity(confidence_info["band"], run_ctx)
        self._reward_history.append(reward_info)

    def _apply_behavior_prediction(self, run_ctx: RunContext, fingerprint: str, step_id: Optional[str]) -> None:
        if not fingerprint:
            return
        reward_trace = run_ctx.get("reward_trace") or []
        recent_rewards = [entry.get("reward", 0.0) for entry in reward_trace[-3:]]
        repair_history = run_ctx.get("repair_events")
        prediction = self.behavior_learner.predict(fingerprint, recent_rewards, repair_history)
        entry = {
            "step_id": step_id,
            "fingerprint": fingerprint,
            **prediction,
        }
        run_ctx.setdefault("behavior_predictions", []).append(entry)
        run_ctx["behavior_difficulty"] = prediction["difficulty"]
        run_ctx["selector_bias"] = prediction["selector_bias"]
        run_ctx["preempt_repair"] = prediction["likely_repair"]
        self._selector_bias = prediction["selector_bias"]
        self._anticipate_repair = prediction["likely_repair"]
        self._behavior_difficulty = prediction["difficulty"]
        self._apply_difficulty_bias(prediction["difficulty"])
        suggestions = run_ctx.setdefault("suggested_subgoals", [])
        for subgoal in prediction.get("recommended_subgoals", []):
            if subgoal not in suggestions:
                suggestions.append(subgoal)

    def _apply_skill_suggestions(
        self,
        run_ctx: RunContext,
        state: Dict[str, Any],
        failure_reason: Optional[str],
        planned_step: StepMeta,
    ) -> None:
        registry = getattr(self, "_skill_registry", None)
        if not registry:
            return
        state_payload = {
            "mode": state.get("mode"),
            "intent": getattr(state.get("intent"), "intent", state.get("intent")),
            "difficulty": self._behavior_difficulty,
            "fingerprint": run_ctx.get("current_fingerprint"),
            "active_subgoal": run_ctx.get("active_plan_target"),
            "missing_fields": state.get("missing_fields") or run_ctx.get("missing_fields"),
            "dom_fingerprint": run_ctx.get("current_fingerprint"),
        }
        failure_payload = {"reason": failure_reason} if failure_reason else None
        suggestions = registry.suggestions(state_payload, failure_payload)
        if not suggestions:
            return
        subgoals = suggestions.get("subgoals") or []
        repairs = suggestions.get("repairs") or []
        if subgoals:
            skill_subgoals = run_ctx.setdefault("suggested_subgoals", [])
            for suggestion in subgoals:
                if suggestion not in skill_subgoals:
                    skill_subgoals.append(suggestion)
        if repairs:
            run_ctx.setdefault("skill_repair_suggestions", []).extend(repairs)
        events = suggestions.get("skills") or []
        if events:
            skill_log = run_ctx.setdefault("skills", [])
            for entry in events:
                skill_log.append({
                    **entry,
                    "step_id": planned_step.get("step_id"),
                    "failure": failure_reason,
                })

    def _apply_difficulty_bias(self, difficulty: float) -> None:
        if difficulty >= 0.8:
            self._recovery_severity = min(3, self._recovery_severity + 1)
        elif difficulty <= 0.3 and self._recovery_severity > 0:
            self._recovery_severity -= 1

    def _update_adaptive_plan(
        self,
        run_ctx: RunContext,
        state: Dict[str, Any],
        dom_snapshot: str,
        failure_reason: Optional[str],
    ) -> None:
        page_intent = state.get("intent")
        if not dom_snapshot or page_intent is None:
            return
        if self._adaptive_plan is None:
            self._adaptive_plan = self.adaptive_planner.build_initial_plan(page_intent, dom_snapshot)
            self._append_trace("adaptive_plan_init", targets=[step.target for step in self._adaptive_plan.steps])
        reward_trace = run_ctx.get("reward_trace") or []
        last_reward_entry = reward_trace[-1] if reward_trace else None
        reward = last_reward_entry.get("reward", 0.0) if last_reward_entry else self._last_reward
        confidence = last_reward_entry.get("confidence") if last_reward_entry else self._confidence_state
        failures = [failure_reason] if failure_reason else None
        repair_events = run_ctx.get("repair_events") or None
        self._adaptive_plan = self.adaptive_planner.update_plan_on_feedback(
            self._adaptive_plan,
            reward,
            confidence,
            failures,
            repair_events,
        )
        strategist_state = {
            "mode": state.get("mode"),
            "severity": self._recovery_severity,
            "suggested_subgoals": run_ctx.get("suggested_subgoals") or [],
        }
        self._adaptive_plan = self.adaptive_planner.expand_with_subgoals(self._adaptive_plan, strategist_state)
        emitted = self.adaptive_planner.emit_next_target(self._adaptive_plan)
        if emitted:
            run_ctx.setdefault("next_targets", []).append(emitted.target)
            run_ctx["active_plan_target"] = emitted.target
            run_ctx.setdefault("planner_events", []).append({
                "type": "subgoal",
                "name": emitted.target,
                "status": "issued",
            })
            if emitted.metadata.get("preferred_selectors"):
                run_ctx.setdefault("memory_hints", []).append({
                    "target": emitted.target,
                    "selectors": emitted.metadata["preferred_selectors"],
                })
        run_ctx.setdefault("plan_evolution", []).append({
            "cursor": self._adaptive_plan.cursor,
            "needs_replan": self._adaptive_plan.needs_replan,
            "targets": list(self._adaptive_plan.state.get("targets", [])),
        })
        if self._adaptive_plan.needs_replan:
            run_ctx["force_replan"] = True
            run_ctx["force_replan_reason"] = "adaptive_planner"

    def _record_memory(self, run_ctx: RunContext, dom_snapshot: str) -> None:
        if not dom_snapshot:
            return
        fingerprint = self._page_fingerprint(run_ctx.get("current_url"), dom_snapshot)
        repairs = []
        for event in run_ctx.get("repair_events") or []:
            patch = event.get("patch") or {}
            selector = patch.get("new_selector") or patch.get("selector")
            if selector:
                repairs.append(selector)
        subgoals = run_ctx.get("next_targets") or []
        reward_trace = run_ctx.get("reward_trace")
        behavior_summary = run_ctx.get("behavior_summary")
        if not behavior_summary:
            behavior_summary = self.behavior_learner.summarize().get(fingerprint)
        self.agent_memory.record(fingerprint, repairs or None, subgoals or None, reward_trace, behavior_summary)
        run_ctx["memory_summary"] = self.agent_memory.summarize_experience()
        hint = self.agent_memory.retrieve(fingerprint)
        if hint:
            run_ctx.setdefault("memory_hints", []).append({"fingerprint": fingerprint, **hint})
            self._apply_memory_hint(hint)

    def _apply_memory_hint(self, hint: Optional[AgentMemoryHint]) -> None:
        if not hint:
            return
        bias = hint.get("selector_bias") if isinstance(hint, dict) else None
        if isinstance(bias, str):
            self._selector_bias = bias
        self.adaptive_planner.set_memory_hint(hint)

    def _page_fingerprint(self, url: Optional[str], dom_snapshot: str) -> str:
        basis = (url or "unknown") + "::" + dom_snapshot[:512]
        return hashlib.md5(basis.encode("utf-8", "ignore")).hexdigest()

    def _attempt_self_repair(self, run_ctx: RunContext, planned_step: StepMeta, dom: str, failure_reason: str | None) -> None:
        if not self.self_repair:
            return
        action_info = {
            "step": planned_step,
            "action_payload": planned_step.get("action_payload", {}),
            "failure": failure_reason,
        }
        patch = self.self_repair.analyze_failure(run_ctx, dom, action_info, self._last_reward, self._confidence_state)
        if not patch:
            return
        self.self_repair.apply_patch_to_strategist(self, patch)
        self.on_repair_applied(patch)
        self.self_repair.record_repair_event(run_ctx, patch, {"step_id": planned_step.get("step_id")})

    def on_failure_detected(self, info: Dict[str, Any]) -> None:
        self._append_trace("failure_detected", **info)

    def on_repair_applied(self, patch: Dict[str, Any]) -> None:
        self._append_trace("repair_applied", patch_type=patch.get("type"), target=patch.get("target_step"))

    def _update_recovery_severity(self, band: str, run_ctx: RunContext) -> None:
        if band == "low":
            self._recovery_severity = min(3, self._recovery_severity + 1)
            if self._recovery_severity >= 2:
                run_ctx["force_replan"] = True
                run_ctx["force_replan_reason"] = "low_confidence"
        elif band == "high":
            self._recovery_severity = max(0, self._recovery_severity - 1)
            if run_ctx.get("force_replan_reason") == "low_confidence":
                run_ctx.pop("force_replan", None)
                run_ctx.pop("force_replan_reason", None)
        run_ctx["recovery_severity"] = self._recovery_severity
        self._append_trace("confidence", band=band, severity=self._recovery_severity, reward=self._last_reward)

    def _heal_selector_via_module(self, planned_step: StepMeta, features: DomFeatures) -> Optional[Dict[str, Any]]:
        dom = self._build_clickable_elements(features)
        if not dom:
            return None
        payload = planned_step.get("action_payload", {})
        broken_selector = payload.get("selector") or ""
        intent = self._build_healing_intent(payload)
        healed = heal_selector(dom, broken_selector, intent=intent)
        if not healed:
            return None
        return {
            "selector": healed.selector,
            "reason": healed.reason,
            "confidence": healed.confidence,
            "original": healed.original_selector,
        }

    def _build_clickable_elements(self, features: DomFeatures) -> List[Dict[str, Any]]:
        elements: List[Dict[str, Any]] = []
        for entry in features.get("buttons", []):
            selector = entry.get("selector") or ""
            if not selector:
                continue
            elements.append({
                "selector": selector,
                "tag": entry.get("tag", "button"),
                "text": entry.get("text", ""),
                "clickable": True,
                "role": entry.get("attributes", {}).get("role", "button"),
            })
        for entry in features.get("links", []):
            selector = entry.get("selector") or ""
            if not selector:
                continue
            elements.append({
                "selector": selector,
                "tag": entry.get("tag", "a"),
                "text": entry.get("text", ""),
                "clickable": True,
                "role": entry.get("attributes", {}).get("role", "link"),
            })
        return elements

    def _build_healing_intent(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text_hint = payload.get("text") or payload.get("label") or payload.get("name")
        role_hint = payload.get("role")
        action = (payload.get("action") or "").lower()
        if not role_hint and action == "click":
            role_hint = "button"
        intent: Dict[str, Any] = {}
        if text_hint:
            intent["text"] = text_hint
        if role_hint:
            intent["role"] = role_hint
        return intent or None

    def should_skip_step(self, run_ctx: RunContext, planned_step: StepMeta) -> bool:
        payload = planned_step.get("action_payload", {})
        action = (payload.get("action") or "").lower()
        if action == "navigate":
            expected = payload.get("url")
            current = run_ctx.get("current_url")
            if expected and current and self._normalize_url(expected) == self._normalize_url(current):
                self._append_trace("skip_rule", rule="redundant_navigation", step_id=planned_step.get("step_id"))
                return True
        if action == "fill" and self._last_features:
            if self._filled_targets(payload, self._last_features):
                self._append_trace("skip_rule", rule="autofill", step_id=planned_step.get("step_id"))
                return True
        if action == "click" and run_ctx.get("recent_transition"):
            source = planned_step.get("source") or "plan"
            if source not in {"cookie_popup", "micro_repair"}:
                self._append_trace("skip_rule", rule="post-transition", step_id=planned_step.get("step_id"))
                return True
        return False

    def should_replan(self, run_ctx: RunContext, planned_step: StepMeta, step_outcome: Dict[str, Any]) -> bool:
        if self._last_mode == "error_page":
            return True
        if run_ctx.get("force_replan"):
            return True
        redirects = run_ctx.get("redirects") or []
        if redirects:
            return True
        meta = step_outcome.get("meta") or {}
        redirect_url = meta.get("redirect_url")
        if redirect_url and redirect_url != run_ctx.get("current_url"):
            return True
        if meta.get("missing_fields"):
            return True
        return False

    def should_abort(self) -> bool:
        if self._repair_attempts >= self.failure_budget:
            return True
        return any(count >= self.failure_limit for count in self._failure_counts.values())

    def _schedule_progressive_recovery(self, run_ctx: RunContext, planned_step: StepMeta) -> None:
        step_id = planned_step.get("step_id") or planned_step.get("action_payload", {}).get("id") or "step"
        count = self._recovery_counts.get(step_id, 0) + 1
        self._recovery_counts[step_id] = count
        run_ctx.setdefault("recovery_counts", {})[step_id] = count
        target_url = run_ctx.get("current_url") or planned_step.get("action_payload", {}).get("url")
        if count == 1:
            actions = [{"action": "reload_if_failed", "name": f"reload_after_{step_id}"}]
            stage = "reload"
        elif count == 2:
            if target_url:
                actions = [{"action": "navigate", "url": target_url, "name": f"revisit_{step_id}"}]
            else:
                actions = [{"action": "extract_dom", "name": f"snapshot_{step_id}"}]
            stage = "navigate"
        else:
            actions = [{"action": "extract_dom", "name": f"snapshot_{step_id}"}]
            stage = "extract_dom"
            run_ctx["force_replan"] = True
        self.insert_steps(actions, bucket=planned_step.get("bucket"), tag="progressive_recovery")
        self._append_trace("progressive_recovery", stage=stage, step_id=step_id, count=count)

    def record_failure(self, signature: str) -> None:
        key = signature.lower().strip() or "error"
        self._failure_counts[key] = self._failure_counts.get(key, 0) + 1
        self._append_trace("failure", signature=key, count=self._failure_counts[key])

    def _max_failure_count(self) -> int:
        if not self._failure_counts:
            return 0
        return max(self._failure_counts.values())

    def _skip_login_steps(self) -> None:
        pending = self._steps[self._cursor :]
        remaining = []
        skipped = 0
        for step in pending:
            action = (step.get("action") or "").lower()
            selector = step.get("selector") or ""
            if action in {"fill", "click"} and any(token in selector for token in {"user", "pass", "login"}):
                skipped += 1
                continue
            remaining.append(step)
        if skipped:
            self._steps = self._steps[: self._cursor] + remaining
            self._append_trace("login_skip", removed=skipped)

    def _inject_login_flow(self, planned_step: StepMeta) -> None:
        if self._login_flow_triggered:
            return
        actions = [
            {"action": "fill", "selector": "#username", "text": "tomsmith"},
            {"action": "fill", "selector": "#password", "text": "SuperSecretPassword!"},
            {"action": "click", "selector": "button[type='submit']"},
            {"action": "wait_for_navigation", "timeout": 8000},
            {"action": "screenshot", "name": "secure_area.png"},
        ]
        self.insert_steps(actions, bucket=planned_step.get("bucket"), tag="login_flow")
        self._login_flow_triggered = True
        self._append_trace("login_flow", inserted=len(actions))

    def _insert_cookie_dismiss(self, popup: Dict[str, Any], planned_step: StepMeta) -> None:
        selector = popup.get("selector") or "button"
        action = {
            "action": "click",
            "selector": selector,
            "name": "dismiss_cookie",
            "text": popup.get("text"),
        }
        self.insert_steps([action], bucket=planned_step.get("bucket"), tag="cookie_popup")

    def _apply_cookie_popup_patch(self, planned_step: StepMeta, repair: MicroRepair) -> None:
        new_selector = repair.get("new_selector")
        if not new_selector:
            return
        payload = planned_step.setdefault("action_payload", {})
        payload["selector"] = new_selector
        payload["metadata"] = {"reason": repair.get("reason"), "patched": True}
        planned_step["selector"] = new_selector

    def _insert_interference_actions(self, findings: List[Dict[str, Any]], planned_step: StepMeta) -> None:
        actions: List[Dict[str, Any]] = []
        for idx, finding in enumerate(findings, start=1):
            selector = finding.get("selector") if isinstance(finding, dict) else getattr(finding, "selector", "")
            reason = finding.get("reason") if isinstance(finding, dict) else getattr(finding, "reason", "interference")
            text = finding.get("text") if isinstance(finding, dict) else getattr(finding, "text", None)
            if not selector:
                continue
            actions.append({
                "action": "click",
                "selector": selector,
                "name": f"dismiss_interference_{idx}",
                "text": text,
                "metadata": {"reason": reason},
            })
        if actions:
            self.insert_steps(actions, bucket=planned_step.get("bucket"), tag="interference")

    def _queue_repair_step(self, planned_step: StepMeta, repair: MicroRepair) -> None:
        patched = dict(planned_step.get("action_payload", {}))
        patched["selector"] = repair.get("new_selector")
        patched["metadata"] = {"reason": repair.get("reason"), "patched": True}
        payload = dict(patched)
        payload.setdefault("id", f"auto_{next(self._inserted_counter)}")
        step_meta: StepMeta = {
            "step_id": payload["id"],
            "task_id": payload.get("task_id", "micro_repair_task"),
            "bucket": planned_step.get("bucket"),
            "action": payload.get("action"),
            "selector": payload.get("selector"),
            "url": payload.get("url"),
            "action_payload": payload,
            "source": "micro_repair",
        }
        insert_index = self._cursor
        while insert_index < len(self._steps) and self._steps[insert_index].get("source") == "cookie_popup":
            insert_index += 1
        self._steps[insert_index:insert_index] = [step_meta]
        self._append_trace("insert", count=1, tag="micro_repair")

    def _mark_active_subgoal(self, run_ctx: RunContext, status: str) -> None:
        target = run_ctx.get("active_plan_target")
        if not target:
            return
        run_ctx.setdefault("planner_events", []).append({
            "type": "subgoal",
            "name": target,
            "status": status,
        })
        if status in {"completed", "failed"}:
            run_ctx.pop("active_plan_target", None)

    def _update_behavior_model(self, run_ctx: RunContext, dom_snapshot: str) -> None:
        fingerprint = run_ctx.get("current_fingerprint")
        if not fingerprint and dom_snapshot:
            fingerprint = self._page_fingerprint(run_ctx.get("current_url"), dom_snapshot)
        if not fingerprint:
            return
        self.behavior_learner.update(
            fingerprint,
            run_ctx.get("reward_trace"),
            run_ctx.get("planner_events"),
            run_ctx.get("repair_events"),
        )
        summary = self.behavior_learner.summarize().get(fingerprint)
        if summary:
            run_ctx["behavior_summary"] = summary

    def _find_similar_selector(self, selector: str, features: DomFeatures) -> Optional[str]:
        target = selector.strip("#[].'")
        for entry in features.get("buttons", []) + features.get("inputs", []):
            candidate = (entry.get("selector") or "").strip()
            if not candidate:
                continue
            normalized = candidate.strip("#[].'")
            if not normalized:
                continue
            if target in normalized or normalized in target:
                return candidate
            attrs = entry.get("attributes", {})
            for attr_key in ("name", "id", "data-testid"):
                attr_val = (attrs.get(attr_key) or "").strip()
                if attr_val and (target in attr_val or attr_val in target):
                    return entry.get("selector") or self._selector_from_attr(attr_key, attr_val)
        return None

    def _infer_label_from_selector(self, selector: str) -> Optional[str]:
        raw = selector.strip("#[]")
        raw = raw.replace("name=", "").replace("'", "").replace("\"", "")
        if not raw:
            return None
        tokens = raw.split("-")
        if len(tokens) == 1:
            return raw
        return " ".join(tokens)

    def _find_by_label(self, label: Optional[str], features: DomFeatures) -> Optional[str]:
        if not label:
            return None
        label_lower = label.lower()
        for entry in features.get("buttons", []):
            text = (entry.get("text") or "").lower()
            if label_lower in text or text in label_lower:
                return entry.get("selector") or self._selector_from_attr("text", entry.get("text", ""))
        return None

    def _infer_form_selector(self, selector: str, features: DomFeatures) -> Optional[str]:
        target = selector.lower()
        if "pass" in target:
            for entry in features.get("inputs", []):
                if entry.get("type") == "password":
                    return entry.get("selector") or self._selector_from_attr("name", entry.get("name", "password"))
        if any(key in target for key in {"user", "email"}):
            for entry in features.get("inputs", []):
                name = (entry.get("name") or "").lower()
                if any(keyword in name for keyword in {"user", "email"}):
                    return entry.get("selector") or self._selector_from_attr("name", entry.get("name", "user"))
        return None

    def _selector_from_attr(self, attr: str, value: str) -> str:
        if attr == "id" and value:
            return f"#{value}"
        if attr == "name" and value:
            return f"[name='{value}']"
        if attr == "text" and value:
            return f"button:contains('{value}')"
        return value

    def _filled_targets(self, payload: Dict[str, Any], features: DomFeatures) -> bool:
        fields = payload.get("fields") or []
        selectors = [field.get("selector") for field in fields if field.get("selector")]
        if payload.get("selector"):
            selectors.append(payload.get("selector"))
        for selector in selectors:
            if selector and selector in features.get("filled_inputs", {}):
                return True
        return False

    def _normalize_url(self, url: str) -> str:
        return (url or "").split("?")[0].rstrip("/").lower()

    def _load_dom_snapshot(self, step_outcome: Dict[str, Any]) -> str:
        dom = step_outcome.get("dom_snapshot")
        if dom:
            return dom
        artifact_path = step_outcome.get("failure_dom_path")
        if not artifact_path:
            return ""
        try:
            return Path(artifact_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            self._append_trace("artifact_missing", path=str(artifact_path))
            return ""

    def _collect_failure_artifacts(self, step_outcome: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        dom_path = step_outcome.get("failure_dom_path")
        screenshot_path = step_outcome.get("failure_screenshot_path")
        if not dom_path and not screenshot_path:
            return None
        return {"dom_path": dom_path, "screenshot_path": screenshot_path}

    def learn_from_past(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        states = batch.get("states") or []
        summary = {
            "states": 0,
            "memory_updates": 0,
            "improved_subgoals": 0,
            "bias_updates": 0,
            "skill_events": 0,
            "learned": [],
        }
        for state in states:
            fingerprint = state.get("fingerprint")
            if not fingerprint:
                continue
            reward_trace = state.get("reward_trace") or []
            planner_events = state.get("planner_events") or []
            repair_events = state.get("repair_events") or []
            self.behavior_learner.update(fingerprint, reward_trace, planner_events, repair_events)
            prediction = self.behavior_learner.predict(fingerprint, [entry.get("reward", 0.0) for entry in reward_trace[-3:]], repair_events)
            selectors = self._selectors_from_repairs(repair_events)
            subgoals = self._merge_offline_subgoals(state, prediction)
            behavior_summary = state.get("behavior_summary") or {"last_prediction": prediction}
            stability_summary = state.get("stability") if isinstance(state.get("stability"), dict) else None
            self.agent_memory.record(fingerprint, selectors, subgoals, reward_trace, behavior_summary, stability_summary)
            summary["states"] += 1
            summary["memory_updates"] += 1
            summary["improved_subgoals"] += len(subgoals)
            if prediction.get("selector_bias"):
                self._selector_bias = prediction["selector_bias"]
                summary["bias_updates"] += 1
            self._anticipate_repair = bool(prediction.get("likely_repair"))
            summary["skill_events"] += self._record_skill_feedback(state, bool(subgoals))
            summary["learned"].append({"fingerprint": fingerprint, "subgoals": subgoals})
        summary["skill_stats"] = SkillRegistry.stats()
        summary["batch_tag"] = batch.get("tag")
        return summary

    def _selectors_from_repairs(self, repairs: Sequence[Dict[str, Any]]) -> List[str]:
        selectors: List[str] = []
        for event in repairs or []:
            patch = event.get("patch") or {}
            selector = patch.get("new_selector") or patch.get("selector")
            if selector:
                selectors.append(selector)
        return selectors

    def _merge_offline_subgoals(self, state: Dict[str, Any], prediction: Dict[str, Any]) -> List[str]:
        candidates: List[str] = []
        for key in ("alternate_subgoals", "suggested_subgoals"):
            for entry in state.get(key) or []:
                if isinstance(entry, str):
                    candidates.append(entry)
        for entry in prediction.get("recommended_subgoals", []):
            if isinstance(entry, str):
                candidates.append(entry)
        return list(dict.fromkeys(candidates))

    def _record_skill_feedback(self, state: Dict[str, Any], success: bool) -> int:
        events = state.get("skill_events") or []
        count = 0
        for event in events:
            SkillRegistry.record_feedback(event.get("name", "skill"), success=success, metadata={
                "step_id": state.get("step_id"),
                "failure": state.get("failure_reason"),
            })
            count += 1
        if not events and state.get("failure_reason"):
            SkillRegistry.record_feedback("replay_analyzer", success=success, metadata={"failure": state.get("failure_reason")})
            count += 1
        return count


__all__ = ["StrategistV2", "StrategistBase"]
