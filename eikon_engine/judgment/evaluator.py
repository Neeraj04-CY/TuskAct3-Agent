from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from eikon_engine.capabilities.models import CapabilityRequirement
from eikon_engine.missions.mission_schema import MissionSpec, MissionSubgoal
from eikon_engine.missions.models import SafetyContract
from eikon_engine.learning.index import LearningBias


JudgmentDecisionType = str  # "allow" | "halt" | "request_approval"


@dataclass
class JudgmentDecision:
    decision: JudgmentDecisionType
    explanation: str
    risk_factors: List[str]
    confidence: float

    def to_payload(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "explanation": self.explanation,
            "risk_factors": list(self.risk_factors),
            "confidence": self.confidence,
        }


class JudgmentEvaluator:
    """Lightweight, data-driven judgment layer for pre-action risk gating."""

    irreversible_actions = {"submit_form", "download_file", "execute_script", "delete_resource", "upload_file"}
    identity_markers = ("imperson", "identity", "authority", "admin", "elevated")

    def evaluate(
        self,
        *,
        mission_spec: MissionSpec,
        subgoal: MissionSubgoal,
        capability_requirements: Sequence[CapabilityRequirement],
        safety_contract: Optional[SafetyContract],
        learning_bias: Optional[LearningBias],
        predicted_actions: Iterable[str] | None = None,
        page_intent: Optional[str] = None,
    ) -> JudgmentDecision:
        factors: List[str] = []

        intent_label = self._normalize_page_intent(page_intent)

        if self._requires_identity(capability_requirements, subgoal, mission_spec):
            factors.append("identity_or_authority_required")
        if self._is_irreversible(predicted_actions, subgoal):
            factors.append("irreversible_action_detected")
        if self._breaches_safety_contract(predicted_actions, safety_contract):
            factors.append("safety_contract_block")
        if intent_label and intent_label.lower().startswith("login"):
            factors.append("page_intent_login")
        if learning_bias and getattr(learning_bias, "confidence", None) is not None:
            try:
                if float(getattr(learning_bias, "confidence")) < 0.2:
                    factors.append("learning_low_confidence")
            except Exception:
                pass

        if not factors:
            return JudgmentDecision(decision="allow", explanation="No blocking risk factors detected", risk_factors=[], confidence=0.9)

        if "identity_or_authority_required" in factors or "irreversible_action_detected" in factors:
            return JudgmentDecision(
                decision="halt",
                explanation=self._build_explanation(subgoal, factors),
                risk_factors=factors,
                confidence=0.8,
            )

        return JudgmentDecision(
            decision="request_approval",
            explanation=self._build_explanation(subgoal, factors),
            risk_factors=factors,
            confidence=0.6,
        )

    def _normalize_page_intent(self, page_intent: Any | None) -> Optional[str]:
        if page_intent is None:
            return None
        if isinstance(page_intent, str):
            return page_intent
        if isinstance(page_intent, dict):
            for key in ("intent", "page_intent", "label", "type"):
                value = page_intent.get(key)
                if isinstance(value, str):
                    return value
        if isinstance(page_intent, (list, tuple)):
            for item in page_intent:
                normalized = self._normalize_page_intent(item)
                if normalized:
                    return normalized
        try:
            return str(page_intent)
        except Exception:
            return None

    def _requires_identity(
        self,
        capability_requirements: Sequence[CapabilityRequirement],
        subgoal: MissionSubgoal,
        mission_spec: MissionSpec,
    ) -> bool:
        text_blobs = [mission_spec.instruction, subgoal.description]
        for req in capability_requirements:
            text_blobs.append(req.capability_id)
        lowered = " ".join(text_blobs).lower()
        return any(marker in lowered for marker in self.identity_markers)

    def _is_irreversible(self, predicted_actions: Iterable[str] | None, subgoal: MissionSubgoal) -> bool:
        actions = {action.lower() for action in (predicted_actions or [])}
        description = subgoal.description.lower()
        if any(marker in description for marker in ("submit", "delete", "upload", "transfer")):
            actions.add("submit_form")
        return any(action in self.irreversible_actions for action in actions)

    def _breaches_safety_contract(
        self,
        predicted_actions: Iterable[str] | None,
        safety_contract: Optional[SafetyContract],
    ) -> bool:
        if not safety_contract:
            return False
        blocked = set(safety_contract.blocked_actions or [])
        if not blocked:
            return False
        actions = {action.lower() for action in (predicted_actions or [])}
        return bool(blocked.intersection(actions or set()))

    def _build_explanation(self, subgoal: MissionSubgoal, factors: List[str]) -> str:
        detected = ", ".join(sorted(factors)) or "none"
        return (
            f"Detected risk factors ({detected}) for '{subgoal.description}'. "
            "Continuing could perform irreversible or authority-bound actions; halting until reviewed."
        )


__all__ = ["JudgmentEvaluator", "JudgmentDecision"]