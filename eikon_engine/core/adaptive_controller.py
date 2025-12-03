"""Adaptive loop controller that coordinates LLM-powered plan repairs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, MutableSequence

from eikon_engine.api.llm_repair import RepairResponse, request_llm_fix
from eikon_engine.browser.schema_v1 import FailureReport, StepAction

PlanDict = Dict[str, Any]


@dataclass
class AdaptiveController:
    """Tracks failure patterns and applies plan deltas safely."""

    max_corrections: int = 3
    max_selector_repairs: int = 2
    corrections: int = field(init=False, default=0)
    selector_repairs: int = field(init=False, default=0)
    _failure_signatures: set[str] = field(init=False, default_factory=set)

    def should_call_llm(self, failure_report: FailureReport) -> bool:
        """Return True when we still have budget and failure is new."""

        if self.corrections >= self.max_corrections:
            return False
        signature = self._signature(failure_report)
        if signature in self._failure_signatures:
            return False
        return bool(failure_report.get("error"))

    def propose_fix(self, failure_report: FailureReport) -> RepairResponse | None:
        """Request a repair suggestion from the LLM helper."""

        if not self.should_call_llm(failure_report):
            return None
        response = request_llm_fix(failure_report)
        if not response:
            return None
        if response["type"] == "patch_selector" and self.selector_repairs >= self.max_selector_repairs:
            return None
        self._failure_signatures.add(self._signature(failure_report))
        return response

    def apply_fix(self, plan: PlanDict, delta: RepairResponse | None) -> PlanDict:
        """Mutate the plan according to the repair delta."""

        if not delta:
            return plan
        handler = getattr(self, f"_apply_{delta['type']}", None)
        if not handler:
            return plan
        payload = delta.get("payload") or {}
        handler(plan, payload)
        self.corrections += 1
        if delta["type"] == "patch_selector":
            self.selector_repairs += 1
        return plan

    # --- Delta handlers -------------------------------------------------

    def _apply_replace_step(self, plan: PlanDict, payload: Dict[str, Any]) -> None:
        step_id = payload.get("step_id")
        new_action = payload.get("action") or {}
        if not step_id or not new_action:
            return
        block = self._find_step(plan, step_id)
        if not block:
            return
        actions, index = block
        new_action.setdefault("id", step_id)
        actions[index] = new_action

    def _apply_insert_steps(self, plan: PlanDict, payload: Dict[str, Any]) -> None:
        before_step = payload.get("before_step")
        additions: List[StepAction] = list(payload.get("actions") or [])
        if not additions:
            return
        target_block = self._find_step(plan, before_step) if before_step else None
        if target_block:
            actions, index = target_block
        else:
            actions = self._default_action_block(plan)
            index = len(actions)
        for offset, action in enumerate(additions):
            actions.insert(index + offset, self._with_step_id(plan, action))

    def _apply_patch_selector(self, plan: PlanDict, payload: Dict[str, Any]) -> None:
        step_id = payload.get("step_id")
        if not step_id:
            return
        block = self._find_step(plan, step_id)
        if not block:
            return
        actions, index = block
        selector = payload.get("selector")
        value = payload.get("value")
        if selector:
            actions[index]["selector"] = selector
        if value is not None:
            actions[index]["value"] = value

    def _apply_navigate(self, plan: PlanDict, payload: Dict[str, Any]) -> None:
        url = payload.get("url")
        if not url:
            return
        navigate_step: StepAction = {
            "action": "navigate",
            "url": url,
        }
        before_step = payload.get("before_step")
        self._apply_insert_steps(
            plan,
            {
                "before_step": before_step,
                "actions": [navigate_step],
            },
        )

    # --- Helpers --------------------------------------------------------

    def _find_step(self, plan: PlanDict, step_id: str | None) -> tuple[MutableSequence[StepAction], int] | None:
        if not step_id:
            return None
        for task in plan.get("tasks", []):
            actions: MutableSequence[StepAction] = task.get("inputs", {}).get("actions", [])
            for index, action in enumerate(actions):
                if action.get("id") == step_id:
                    return actions, index
        actions = plan.get("actions") or []
        for index, action in enumerate(actions):
            if action.get("id") == step_id:
                return actions, index
        return None

    def _default_action_block(self, plan: PlanDict) -> MutableSequence[StepAction]:
        if plan.get("tasks"):
            return plan["tasks"][0].setdefault("inputs", {}).setdefault("actions", [])
        return plan.setdefault("actions", [])

    def _with_step_id(self, plan: PlanDict, action: StepAction) -> StepAction:
        if action.get("id"):
            return action
        action = dict(action)
        action["id"] = self._generate_step_id(plan)
        return action

    def _generate_step_id(self, plan: PlanDict) -> str:
        existing = set()
        for task in plan.get("tasks", []):
            for action in task.get("inputs", {}).get("actions", []):
                if action.get("id"):
                    existing.add(action["id"])
        for action in plan.get("actions", []) or []:
            if action.get("id"):
                existing.add(action["id"])
        index = 1
        while True:
            candidate = f"s{index}"
            if candidate not in existing:
                return candidate
            index += 1

    def _signature(self, failure_report: FailureReport) -> str:
        dom_excerpt = failure_report.get("dom_excerpt", "")[:120]
        return f"{failure_report.get('step_id')}|{failure_report.get('error')}|{dom_excerpt}"

```