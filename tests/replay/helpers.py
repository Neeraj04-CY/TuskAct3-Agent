from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def seed_autonomy_run(root: Path, name: str = "run_seed") -> Path:
    run_dir = root / "artifacts" / "autonomy" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "goal": "Offline improvement smoke test",
        "completed": False,
        "reason": "replay seed",
    }
    run_ctx: Dict[str, object] = {
        "current_url": "https://example.com/login",
        "reward_trace": [
            {
                "step_id": "step_001",
                "reward": 0.0,
                "reasons": ["no_dom_change"],
                "confidence": {"band": "low", "confidence": 0.2},
            }
        ],
        "repair_events": [
            {
                "patch": {"selector": "#email", "reason": "label_match"},
                "step_id": "step_001",
            }
        ],
        "planner_events": [
            {"type": "subgoal", "name": "login_form", "status": "failed"}
        ],
        "plan_evolution": [
            {"cursor": 0, "needs_replan": True, "targets": ["login_form"]}
        ],
        "behavior_summary": {
            "last_prediction": {
                "difficulty": 0.8,
                "selector_bias": "css",
                "recommended_subgoals": ["reset_session"],
            }
        },
        "behavior_difficulty": 0.8,
        "skills": [
            {
                "name": "login",
                "subgoals": 1,
                "repairs": 1,
                "metadata": {"recommended_subgoals": ["capture_session"]},
            }
        ],
        "skill_repair_suggestions": [{"action": "reset_session"}],
        "suggested_subgoals": ["fix_login"],
        "behavior_predictions": [
            {
                "step_id": "step_001",
                "fingerprint": "abc123",
                "difficulty": 0.8,
                "selector_bias": "css",
                "likely_repair": True,
            }
        ],
        "current_fingerprint": "abc123",
    }
    stability = {
        "metrics": {
            "repeated_failures": {"login": 2},
            "reward_drift": 0.2,
            "confidence_delta": 0.15,
            "dom_similarity_prev": 0.8,
        }
    }
    result = {
        "goal": summary["goal"],
        "steps": [
            {
                "step": {
                    "step_id": "step_001",
                    "task_id": "task_auth",
                    "bucket": "auth",
                    "action": "click",
                },
                "result": {
                    "dom_snapshot": "<html>state</html>",
                    "completion": {"complete": False, "reason": "selector_missing"},
                    "error": "selector_missing",
                },
            }
        ],
        "run_context": {**run_ctx, "stability_summary": stability},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (run_dir / "stability_report.json").write_text(json.dumps(stability), encoding="utf-8")
    return run_dir
