from __future__ import annotations

from run_autonomy_demo import build_autonomy_summary, collect_guardrail_blocks


def make_result() -> dict:
    return {
        "goal": "demo",
        "completion": {"complete": True, "reason": "done"},
        "duration_seconds": 4.2,
        "step_count": 3,
        "run_context": {
            "page_intents": [{"intent": "form_entry", "confidence": 0.8}],
            "redirects": [{"from": "a", "to": "b"}],
        },
        "strategist_trace": [{"event": "progressive_recovery", "stage": "reload"}],
        "steps": [
            {
                "step": {"step_id": "s1", "action": "click"},
                "result": {
                    "steps": [
                        {"id": "s1", "status": "blocked", "block_reason": "click_blocked_risky"},
                    ]
                },
            }
        ],
    }


def test_collect_guardrail_blocks_extracts_reasons() -> None:
    blocks = collect_guardrail_blocks(make_result())
    assert blocks
    assert blocks[0]["reason"] == "click_blocked_risky"


def test_build_autonomy_summary_surface_intents_and_events() -> None:
    summary = build_autonomy_summary(make_result())
    assert summary["completed"] is True
    assert summary["page_intents"][0]["intent"] == "form_entry"
    assert summary["guardrail_blocks"]
    assert summary["interventions"]
    assert summary["reward_trace"] == []
    assert summary["repair_events"] == []
