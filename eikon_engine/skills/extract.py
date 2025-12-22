from __future__ import annotations

from typing import Any, Dict, List

from .base import Skill


class ExtractSkill(Skill):
    name = "extract"
    description = "Suggests DOM extraction subgoals for dashboard/reporting pages."

    async def execute(self, context):
        return {
            "status": "noop",
            "skill": self.name,
            "context": list((context or {}).keys()),
        }

    def suggest_subgoals(self, state: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        state = state or {}
        mode = (state.get("mode") or "").lower()
        if mode not in {"dashboard_page", "report", "summary"}:
            return []
        targets = state.get("extract_targets") or [".widget", "table"]
        return [
            {
                "name": "capture_dashboard",
                "type": "extract_dom",
                "selectors": targets,
                "reason": "ExtractSkill detected dashboard-like state",
            }
        ]

    def suggest_repairs(self, failure: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        return []


__all__ = ["ExtractSkill"]
