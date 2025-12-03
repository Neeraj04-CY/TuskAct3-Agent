"""Lightweight multi-step planner with feedback hooks."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from eikon_engine.strategist.agent_memory import AgentMemoryHint


@dataclass
class PlanStep:
    id: str
    target: str
    priority: float = 1.0
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanState:
    steps: List[PlanStep] = field(default_factory=list)
    cursor: int = 0
    needs_replan: bool = False
    state: Dict[str, Any] = field(default_factory=lambda: {"targets": [], "history": [], "updated_patches": []})


class PlannerV1:
    def __init__(self, memory_hint: Optional[AgentMemoryHint] = None) -> None:
        self._state = PlanState()
        self._id_counter = itertools.count(1)
        self._memory_hint = memory_hint

    @property
    def plan(self) -> PlanState:
        return self._state

    def set_memory_hint(self, hint: Optional[AgentMemoryHint]) -> None:
        self._memory_hint = hint
        self._apply_memory_hint()

    def build_initial_plan(self, page_intent: Any, dom_snapshot: str) -> PlanState:
        intent_name = getattr(page_intent, "intent", "unknown")
        targets = self._seed_targets(intent_name)
        steps = [PlanStep(id=f"plan_{next(self._id_counter)}", target=target, priority=1.0) for target in targets]
        self._state = PlanState(steps=steps, state={"targets": list(targets), "history": [], "updated_patches": []})
        self._apply_memory_hint()
        return self._state

    def update_plan_on_feedback(
        self,
        plan: Optional[PlanState],
        reward: float,
        confidence: Optional[Dict[str, Any]],
        failures: List[str] | None,
        repair_events: List[Dict[str, Any]] | None,
    ) -> PlanState:
        plan = plan or self._state
        plan.state.setdefault("history", []).append({"reward": reward, "confidence": confidence})
        low_conf_history = [entry for entry in plan.state["history"][-3:] if (entry.get("confidence") or {}).get("band") == "low"]
        if len(low_conf_history) >= 3:
            plan.needs_replan = True
        if repair_events and len(repair_events) >= 2:
            plan.needs_replan = True
            plan.state.setdefault("updated_patches", []).extend(repair_events[-2:])
        if failures:
            plan.state.setdefault("failures", []).extend(failures)
        self._state = plan
        return plan

    def expand_with_subgoals(self, plan: Optional[PlanState], strategist_state: Dict[str, Any]) -> PlanState:
        plan = plan or self._state
        if plan.needs_replan:
            return plan
        additional_targets = strategist_state.get("suggested_subgoals") or []
        for target in additional_targets:
            plan.steps.append(PlanStep(id=f"plan_{next(self._id_counter)}", target=target, priority=0.8))
            plan.state.setdefault("targets", []).append(target)
        self._state = plan
        return plan

    def emit_next_target(self, plan: Optional[PlanState]) -> Optional[PlanStep]:
        plan = plan or self._state
        while plan.cursor < len(plan.steps):
            step = plan.steps[plan.cursor]
            if step.status == "completed":
                plan.cursor += 1
                continue
            plan.cursor += 1
            step.status = "issued"
            return step
        return None

    def _seed_targets(self, intent_name: str) -> List[str]:
        if intent_name == "auth_form":
            return ["fill_credentials", "submit_login", "verify_landing"]
        if intent_name == "form_entry":
            return ["collect_inputs", "review_summary"]
        if intent_name == "search_results":
            return ["refine_query", "open_top_result"]
        return ["inspect_page", "discover_actions"]

    def _apply_memory_hint(self) -> None:
        if not self._memory_hint:
            return
        selectors = self._memory_hint.get("selectors", [])
        if selectors and self._state.steps:
            self._state.steps[0].metadata["preferred_selectors"] = selectors


__all__ = ["PlannerV1", "PlanState", "PlanStep"]
