"""Self-repair engine for Strategist V2."""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional


FailureReason = Dict[str, Any]
PatchPayload = Dict[str, Any]


class SelfRepairEngine:
    def __init__(
        self,
        *,
        max_attempts_per_step: int = 2,
        max_attempts_global: int = 8,
    ) -> None:
        self.max_attempts_per_step = max_attempts_per_step
        self.max_attempts_global = max_attempts_global
        self._step_attempts: Dict[str, int] = {}
        self._global_attempts = 0
        self._event_log: List[Dict[str, Any]] = []
        self._context: Dict[str, Any] | None = None
        self._patch_counter = itertools.count(1)

    @property
    def event_log(self) -> List[Dict[str, Any]]:
        return list(self._event_log)

    def reset(self) -> None:
        self._step_attempts.clear()
        self._global_attempts = 0
        self._event_log.clear()
        self._context = None
        self._patch_counter = itertools.count(1)

    def analyze_failure(
        self,
        run_ctx: Dict[str, Any],
        last_dom: str,
        last_action: Dict[str, Any],
        reward: float,
        confidence: Optional[Dict[str, Any]],
    ) -> Optional[PatchPayload]:
        if self._global_attempts >= self.max_attempts_global:
            return None
        step_meta = last_action.get("step") or {}
        step_id = step_meta.get("step_id") or last_action.get("step_id") or "unknown"
        if self._step_attempts.get(step_id, 0) >= self.max_attempts_per_step:
            return None
        reason = self._detect_failure_reason(run_ctx, last_dom, last_action, reward, confidence)
        if not reason:
            return None
        self._context = {
            "run_ctx": run_ctx,
            "last_action": last_action,
            "reward": reward,
            "confidence": confidence or {"confidence": 0.5, "band": "medium"},
            "reason": reason,
        }
        patch = self.generate_repair_patch(reason)
        if not patch:
            return None
        patch.setdefault("target_step", step_id)
        patch.setdefault("metadata", {})
        patch["metadata"].setdefault("reason", reason["kind"])
        patch.setdefault("action_payload", last_action.get("action_payload") or step_meta.get("action_payload") or {})
        patch.setdefault("bucket", step_meta.get("bucket"))
        patch.setdefault("id", f"repair_{next(self._patch_counter)}")
        self._step_attempts[step_id] = self._step_attempts.get(step_id, 0) + 1
        self._global_attempts += 1
        return patch

    def generate_repair_patch(self, failure_reason: FailureReason) -> Optional[PatchPayload]:
        context = self._context or {}
        action_payload = (context.get("last_action") or {}).get("action_payload", {})
        selector = action_payload.get("selector", "")
        kind = failure_reason.get("kind")
        if kind == "selector_not_found":
            new_selector = self._widen_selector(selector)
            return {
                "type": "selector_update",
                "new_selector": new_selector,
            }
        if kind == "element_not_clickable":
            return {
                "type": "node_matcher",
                "injection": {"action": "wait_for_selector", "selector": selector or "button", "timeout": 1500},
            }
        if kind == "null_dom_changes":
            return {"type": "recovery_stage", "stage": "navigate"}
        if kind == "low_confidence":
            return {"type": "strategy_param", "param": "confidence_mode", "value": "escalate"}
        if kind == "reward_stagnation":
            subgoal = failure_reason.get("proposal") or f"discover alternate path for {action_payload.get('selector', 'page')}"
            return {"type": "subgoal", "subgoal": subgoal}
        return None

    def apply_patch_to_strategist(self, strategist: Any, patch: PatchPayload) -> None:
        patch_type = patch.get("type")
        if patch_type == "selector_update":
            payload = dict(patch.get("action_payload") or {})
            payload["selector"] = patch.get("new_selector")
            payload.setdefault("action", payload.get("action") or "click")
            strategist.insert_steps([payload], bucket=patch.get("bucket"), tag="micro_repair")
        elif patch_type == "node_matcher":
            wait_action = dict(patch.get("injection") or {})
            wait_action.setdefault("action", "wait_for_selector")
            strategist.insert_steps([wait_action], bucket=patch.get("bucket"), tag="self_repair_wait")
        elif patch_type == "recovery_stage":
            strategist._recovery_severity = max(strategist._recovery_severity, 2)  # noqa: SLF001
        elif patch_type == "strategy_param":
            strategist.failure_budget = max(1, strategist.failure_budget - 1)
            strategist.failure_limit = max(1, strategist.failure_limit - 1)
        elif patch_type == "subgoal":
            strategist.queue_subgoal(patch.get("subgoal", "stabilize_flow"))

    def record_repair_event(self, run_ctx: Dict[str, Any], patch: PatchPayload, details: Dict[str, Any] | None) -> None:
        entry = {
            "patch": {k: v for k, v in patch.items() if k not in {"action_payload"}},
        }
        if details:
            entry.update(details)
        run_ctx.setdefault("repair_events", []).append(entry)
        self._event_log.append(entry)

    def _detect_failure_reason(
        self,
        run_ctx: Dict[str, Any],
        last_dom: str,
        last_action: Dict[str, Any],
        reward: float,
        confidence: Optional[Dict[str, Any]],
    ) -> Optional[FailureReason]:
        failure_text = str(last_action.get("failure") or "").lower()
        if any(token in failure_text for token in {"selector", "not found", "missing"}):
            return {"kind": "selector_not_found"}
        if "not clickable" in failure_text or "disabled" in failure_text:
            return {"kind": "element_not_clickable"}
        reward_trace = run_ctx.get("reward_trace") or []
        if reward_trace:
            recent = reward_trace[-3:]
            if recent and all((entry.get("confidence", {}).get("confidence", 1.0) < 0.25) for entry in recent):
                return {"kind": "low_confidence"}
            if len(recent) >= 3:
                rewards = [entry.get("reward", 0.0) for entry in recent]
                if max(rewards) - min(rewards) < 0.05:
                    return {"kind": "reward_stagnation", "proposal": f"branch alternate path after {recent[-1].get('step_id')}"}
            last_reasons = recent[-1].get("reasons", []) if recent else []
            if any("dom_static" in reason for reason in last_reasons):
                return {"kind": "null_dom_changes"}
        confidence_val = (confidence or {}).get("confidence", 0.5)
        if confidence_val < 0.2:
            return {"kind": "low_confidence"}
        if reward < -1.5:
            return {"kind": "null_dom_changes"}
        return None

    def _widen_selector(self, selector: str) -> str:
        if not selector:
            return "button"
        if selector.startswith("#"):
            return selector[1:]
        if selector.startswith("."):
            return selector
        if " " in selector:
            return selector.split(" ")[0]
        return selector + "_alt"


__all__ = ["SelfRepairEngine"]
