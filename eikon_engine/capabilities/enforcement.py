from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Literal, Mapping

from eikon_engine.capabilities.models import Capability, CapabilityRequirement, CapabilityId

CapabilityDecisionType = Literal["allow", "warn_only", "ask_human"]


@dataclass(frozen=True)
class EnforcementContext:
    threshold: float = 0.8
    critical: float = 0.5
    fail_on_missing_capability: bool = False
    auto_approve_capabilities: bool = False


@dataclass(frozen=True)
class CapabilityDecision:
    capability_id: CapabilityId
    decision: CapabilityDecisionType
    confidence: float
    threshold: float
    critical: float
    reason: str
    required: bool
    missing: bool
    subgoal_id: str | None = None
    source: str | None = None

    def as_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["decision_type"] = "capability_enforcement"
        return payload


def _clamp(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _missing(capability_id: CapabilityId, registry: Mapping[str, Capability]) -> bool:
    return capability_id not in registry


def evaluate_capability(
    requirement: CapabilityRequirement,
    context: EnforcementContext,
    *,
    registry: Mapping[str, Capability],
) -> CapabilityDecision:
    confidence = _clamp(requirement.confidence)
    missing = _missing(requirement.capability_id, registry)
    if missing:
        decision: CapabilityDecisionType = "ask_human" if not context.auto_approve_capabilities else "warn_only"
        reason = "capability_missing"
        if context.fail_on_missing_capability and not context.auto_approve_capabilities:
            decision = "ask_human"
        return CapabilityDecision(
            capability_id=requirement.capability_id,
            decision=decision,
            confidence=confidence,
            threshold=context.threshold,
            critical=context.critical,
            reason=reason,
            required=requirement.required,
            missing=True,
            source=requirement.source,
        )

    if confidence < context.critical and not context.auto_approve_capabilities:
        decision = "ask_human"
        reason = "confidence_below_critical"
    elif confidence < context.threshold:
        decision = "warn_only"
        reason = "confidence_below_threshold"
    else:
        decision = "allow"
        reason = "confidence_meets_threshold"
    return CapabilityDecision(
        capability_id=requirement.capability_id,
        decision=decision,
        confidence=confidence,
        threshold=context.threshold,
        critical=context.critical,
        reason=reason,
        required=requirement.required,
        missing=False,
        source=requirement.source,
    )


def evaluate_capabilities(
    requirements: Iterable[CapabilityRequirement],
    context: EnforcementContext,
    *,
    registry: Mapping[str, Capability],
) -> List[CapabilityDecision]:
    return [evaluate_capability(requirement, context, registry=registry) for requirement in requirements]
