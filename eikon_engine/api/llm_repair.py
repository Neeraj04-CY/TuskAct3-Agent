"""LLM-backed repair helpers (placeholder implementation)."""

from __future__ import annotations

from typing import Any, Dict, Literal, TypedDict

from eikon_engine.browser.schema_v1 import FailureReport

RepairType = Literal["replace_step", "insert_steps", "patch_selector", "navigate"]


class RepairResponse(TypedDict, total=False):
    """LLM-proposed delta description consumed by AdaptiveController."""

    type: RepairType
    payload: Dict[str, Any]


def request_llm_fix(failure_report: FailureReport) -> RepairResponse | None:
    """Placeholder LLM call that emits a deterministic suggestion.

    Real deployments should swap this with an OpenAI / Azure call. We return a
    simple selector patch so the adaptive controller has a baseline delta when
    tests do not mock this helper.
    """

    step_id = failure_report.get("step_id")
    if not step_id:
        return None
    return {
        "type": "patch_selector",
        "payload": {
            "step_id": step_id,
            "selector": "#llm-suggested",
        },
    }
