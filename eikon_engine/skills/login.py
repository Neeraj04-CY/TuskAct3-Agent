from __future__ import annotations

from typing import Any, Dict, List

from .base import SkillBase


class LoginSkill(SkillBase):
    name = "login"
    description = "Ensures login steps are prioritized when auth gates are detected."

    def suggest_subgoals(self, state: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        state = state or {}
        mode = (state.get("mode") or "").lower()
        intent = state.get("intent")
        intent_name = getattr(intent, "intent", intent)
        if mode not in {"login_page", "auth_gate"} and intent_name not in {"login", "auth"}:
            return []
        return [
            {
                "name": "complete_login",
                "type": "auth",
                "reason": "LoginSkill detected authentication gate",
                "selectors": state.get("login_selectors") or ["input[name='username']", "input[name='password']"],
            }
        ]

    def suggest_repairs(self, failure: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        failure_text = (failure or {}).get("reason", "").lower()
        if any(token in failure_text for token in {"unauthorized", "login", "credential"}):
            return [
                {
                    "action": "reset_session",
                    "reason": "Credentials rejected, requesting new session",
                }
            ]
        return []


__all__ = ["LoginSkill"]
