from __future__ import annotations

from typing import Any, Dict, List, Optional

from .registry import SKILL_REGISTRY, get_skill


class SkillRegistry:
    """Shim that exposes legacy registry APIs for backward compatibility."""

    _skill_feedback: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def _skill_instances(cls) -> List[Any]:
        return list(SKILL_REGISTRY.values())

    @classmethod
    def get_all(cls) -> List[Any]:
        return cls._skill_instances()

    @classmethod
    def suggestions(cls, state: Dict[str, Any] | None = None, failure: Dict[str, Any] | None = None) -> Dict[str, Any]:
        skills_meta = []
        for name in SKILL_REGISTRY:
            skills_meta.append({"name": name, "metadata": {"state_keys": list((state or {}).keys())}})
        return {
            "subgoals": [{"skill": name, "reason": "memory_hint"} for name in SKILL_REGISTRY],
            "repairs": [{"skill": name, "reason": "retry"} for name in SKILL_REGISTRY],
            "skills": skills_meta,
        }

    @classmethod
    def record_feedback(cls, name: str, *, success: bool, metadata: Optional[Dict[str, Any]] = None) -> None:
        entry = cls._skill_feedback.setdefault(name, {"success": 0, "fail": 0, "notes": []})
        key = "success" if success else "fail"
        entry[key] += 1
        if metadata:
            entry["notes"].append(metadata)

    @classmethod
    def stats(cls) -> Dict[str, Any]:
        return {name: {"success": data["success"], "fail": data["fail"], "notes": list(data["notes"])} for name, data in cls._skill_feedback.items()}

    @classmethod
    def load(cls, name: str):
        return get_skill(name)


__all__ = ["SkillRegistry"]
