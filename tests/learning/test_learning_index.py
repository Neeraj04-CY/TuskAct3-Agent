from __future__ import annotations

import json
from pathlib import Path

from eikon_engine.learning.index import LearningIndexCache, infer_mission_type
from eikon_engine.learning.signals import load_skill_signals


def _write_record(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_load_skill_signals_filters_low_confidence(tmp_path: Path) -> None:
    root = tmp_path / "learning_logs"
    high_conf = {
        "mission_id": "m-login-1",
        "mission_type": "login",
        "timestamp": "2026-01-13T12:00:00+00:00",
        "confidence_score": 0.82,
        "skills_used": [
            {"skill_name": "login_form_skill", "success": True, "steps_saved": 3},
            {"skill_name": "listing_extraction_skill", "success": False, "steps_saved": 0},
        ],
    }
    low_conf = {
        "mission_id": "m-login-2",
        "mission_type": "login",
        "timestamp": "2026-01-13T12:05:00+00:00",
        "confidence_score": 0.2,
        "skills_used": [{"skill_name": "login_form_skill", "success": False, "steps_saved": 0}],
    }
    _write_record(root / "high.json", high_conf)
    _write_record(root / "low.json", low_conf)

    signals = load_skill_signals(root)

    assert len(signals) == 2  # login + listing skill entries
    login_signal = next(signal for signal in signals if signal.skill_name == "login_form_skill")
    assert login_signal.attempts == 1
    assert login_signal.success_rate == 1.0
    assert round(login_signal.avg_steps_saved, 2) == 3.0


def test_learning_index_cache_detects_file_changes(tmp_path: Path) -> None:
    root = tmp_path / "learning_logs"
    _write_record(
        root / "login.json",
        {
            "mission_id": "m-login-1",
            "mission_type": "login",
            "timestamp": "2026-01-13T12:00:00+00:00",
            "confidence_score": 0.9,
            "skills_used": [{"skill_name": "login_form_skill", "success": True, "steps_saved": 2}],
        },
    )
    cache = LearningIndexCache(root=root)

    login_bias = cache.bias_for_goal("Log into the sample site")
    assert login_bias is not None
    assert login_bias.preferred_skills == ["login_form_skill"]
    assert infer_mission_type("extract listings") == "listing"

    _write_record(
        root / "listing.json",
        {
            "mission_id": "m-listing-1",
            "mission_type": "listing",
            "timestamp": "2026-01-13T13:00:00+00:00",
            "confidence_score": 0.75,
            "skills_used": [{"skill_name": "listing_extraction_skill", "success": True, "steps_saved": 4}],
        },
    )

    listing_bias = cache.bias_for_goal("Find startup listings")
    assert listing_bias is not None
    assert listing_bias.preferred_skills[0] == "listing_extraction_skill"
    metadata = listing_bias.metadata_for("listing_extraction_skill")
    assert metadata and metadata["signal"]["success_rate"] > 0.5
