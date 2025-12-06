from __future__ import annotations

from eikon_engine.replay.curriculum_builder import CurriculumBuilder


def test_curriculum_builder_groups_runs() -> None:
    runs = [
        {"result": {"run_context": {"behavior_difficulty": 0.8}}, "stability": {"metrics": {"repeated_failures": {}, "reward_drift": 0.0, "dom_similarity_prev": 0.1}}},
        {"result": {"run_context": {"behavior_difficulty": 0.4}}, "stability": {"metrics": {"repeated_failures": {"login": 2}, "reward_drift": 0.0, "dom_similarity_prev": 0.1}}},
        {"result": {"run_context": {"behavior_difficulty": 0.5}}, "stability": {"metrics": {"repeated_failures": {}, "reward_drift": 0.2, "confidence_delta": 0.2, "dom_similarity_prev": 0.1}}},
        {"result": {"run_context": {"behavior_difficulty": 0.5}}, "stability": {"metrics": {"repeated_failures": {}, "reward_drift": 0.0, "dom_similarity_prev": 0.9}}},
    ]
    builder = CurriculumBuilder(runs)
    tags = [batch["tag"] for batch in builder.get_curriculum()]
    assert "high_difficulty" in tags
    assert "repeated_failures" in tags
    assert "stability_drift" in tags
    assert "dom_similarity" in tags
