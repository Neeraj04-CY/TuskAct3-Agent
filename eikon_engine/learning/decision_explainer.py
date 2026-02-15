from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal

DecisionType = Literal["override", "refusal", "bias_applied"]
FinalResolution = Literal["override_applied", "refused", "bias_only"]


@dataclass(frozen=True)
class LearningDecisionExplanation:
    mission_id: str
    decision_type: DecisionType
    learning_impact_score: float
    confidence_score: float
    triggering_signals: List[Dict[str, object]]
    planner_conflict: bool
    final_resolution: FinalResolution
    summary: str


def build_learning_decision_explanation(**payload: object) -> LearningDecisionExplanation:
    return LearningDecisionExplanation(**payload)  # type: ignore[arg-type]


def write_learning_decision_explanation(directory: Path | str, explanation: LearningDecisionExplanation) -> Path:
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "learning_decision_explanation.json"
    data = asdict(explanation)
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return target


__all__ = [
    "LearningDecisionExplanation",
    "build_learning_decision_explanation",
    "write_learning_decision_explanation",
    "DecisionType",
    "FinalResolution",
]
