from __future__ import annotations

from typing import Any, Dict, List

from .base import Skill


class FormFillSkill(Skill):
    name = "form_fill"
    description = "Completes missing form inputs when the DOM exposes required fields."

    async def execute(self, context):
        return {
            "status": "noop",
            "skill": self.name,
            "context": list((context or {}).keys()),
        }

    def suggest_subgoals(self, state: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        state = state or {}
        missing_fields = state.get("missing_fields") or state.get("form_fields") or []
        if not missing_fields:
            return []
        return [
            {
                "name": "fill_required_fields",
                "type": "form_fill",
                "fields": missing_fields,
                "confidence": 0.8,
                "reason": "FormFillSkill detected missing required fields",
            }
        ]

    def suggest_repairs(self, failure: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        failure_text = (failure or {}).get("reason", "").lower()
        if not failure_text:
            return []
        if "validation" in failure_text or "missing" in failure_text:
            return [
                {
                    "action": "refill_form",
                    "strategy": "retry_with_defaults",
                    "reason": "Validation or missing field failure detected",
                }
            ]
        return []


__all__ = ["FormFillSkill"]
