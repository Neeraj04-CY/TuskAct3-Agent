from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

CapabilityId = str


@dataclass(frozen=True)
class Capability:
    id: CapabilityId
    name: str
    description: str
    skills: List[str]
    domains: List[str]
    risk_level: str  # "low" | "medium" | "high"
    confidence: Optional[float] = None  # optional telemetry hook; not used in Phase 3.0


__all__ = ["Capability", "CapabilityId"]


@dataclass(frozen=True)
class CapabilityRequirement:
    capability_id: CapabilityId
    required: bool
    confidence: float  # 0.0 – 1.0 planner-side certainty
    reason: str
    source: str | None = None  # optional provenance label


@dataclass(frozen=True)
class PlanCapabilityReport:
    required: List[CapabilityRequirement] = field(default_factory=list)
    missing: List[CapabilityRequirement] = field(default_factory=list)
    optional: List[CapabilityRequirement] = field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "low"


__all__.extend([
    "CapabilityRequirement",
    "PlanCapabilityReport",
])
