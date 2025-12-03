from __future__ import annotations

from eikon_engine.strategist_v2.confidence_scorer import score_decision


def test_confidence_low_band() -> None:
    result = score_decision(-2.0, {"mode": "error"}, 0)
    assert result["band"] == "low"


def test_confidence_medium_band() -> None:
    result = score_decision(0.2, {"mode": "progress"}, 1)
    assert result["band"] == "medium"


def test_confidence_high_band() -> None:
    result = score_decision(2.5, {"mode": "success"}, 0)
    assert result["band"] == "high"
