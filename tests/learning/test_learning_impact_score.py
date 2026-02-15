from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from eikon_engine.learning.impact_score import LearningImpactScore
from eikon_engine.learning.signals import SkillSignal

UTC = timezone.utc


def test_learning_score_computed_and_persisted(tmp_path: Path) -> None:
    signal = SkillSignal(
        skill_name="login_form_skill",
        mission_type="login",
        attempts=3,
        successes=2,
        total_steps_saved=6,
        last_mission_id="m-login",
        last_timestamp=datetime(2026, 1, 13, 12, 0, tzinfo=UTC),
        confidence_samples=3,
        confidence_mean=0.82,
    )

    scorer = LearningImpactScore(signals=[signal], now=datetime(2026, 1, 13, 13, 0, tzinfo=UTC))
    score_value = scorer.score("login_form_skill", "login", "login")

    assert -1.0 <= score_value <= 1.0
    assert score_value > 0

    path = scorer.persist(tmp_path / "learning_index.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert "scores" in payload
    assert payload["scores"], "scores should be persisted"