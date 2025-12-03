"""Confidence scoring helpers for Strategist V2."""

from __future__ import annotations

from typing import Dict


def score_decision(reward: float, strategist_state: Dict[str, object] | None, failure_count: int) -> Dict[str, object]:
    state = strategist_state or {}
    if reward < -1.0 or failure_count > 2:
        band = "low"
    elif reward > 1.0:
        band = "high"
    else:
        band = "medium"
    confidence = max(0.0, min(1.0, (reward + 2.0) / 4.0))
    return {"confidence": round(confidence, 3), "band": band, "state": state}


__all__ = ["score_decision"]
