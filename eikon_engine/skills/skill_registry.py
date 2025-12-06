from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, List, Optional, Sequence, Type

from .base import SkillBase


class SkillRegistry:
    """Autoloads skills and exposes lightweight suggestion helpers."""

    _initialized: bool = False
    _skills: List[SkillBase] = []
    _skill_feedback: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def _ensure_initialized(cls) -> None:
        if cls._initialized:
            return
        cls._initialized = True
        package = importlib.import_module("eikon_engine.skills")
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            if module_name.startswith("_") or module_name in {"base", "skill_registry"}:
                continue
            importlib.import_module(f"{package.__name__}.{module_name}")
        cls._skills = [skill_cls() for skill_cls in cls._skill_classes()]

    @classmethod
    def _skill_classes(cls) -> Sequence[Type[SkillBase]]:
        result: List[Type[SkillBase]] = []
        stack = list(SkillBase.__subclasses__())
        while stack:
            entry = stack.pop()
            result.append(entry)
            stack.extend(entry.__subclasses__())
        return result

    @classmethod
    def get_all(cls) -> List[SkillBase]:
        cls._ensure_initialized()
        return list(cls._skills)

    @classmethod
    def suggestions(cls, state: Dict[str, Any] | None = None, failure: Dict[str, Any] | None = None) -> Dict[str, Any]:
        cls._ensure_initialized()
        aggregated = {"subgoals": [], "repairs": [], "skills": []}
        for skill in cls._skills:
            try:
                subgoals = skill.suggest_subgoals(state)
            except Exception as exc:  # pragma: no cover - defensive
                subgoals = [{"error": str(exc)}]
            try:
                repairs = skill.suggest_repairs(failure)
            except Exception as exc:  # pragma: no cover - defensive
                repairs = [{"error": str(exc)}]
            aggregated["subgoals"].extend(entry for entry in subgoals or [] if isinstance(entry, dict))
            aggregated["repairs"].extend(entry for entry in repairs or [] if isinstance(entry, dict))
            aggregated["skills"].append({
                "name": skill.name,
                "metadata": skill.metadata(),
                "subgoals": len(subgoals or []),
                "repairs": len(repairs or []),
            })
        return aggregated

    @classmethod
    def record_feedback(cls, name: str, *, success: bool, metadata: Optional[Dict[str, Any]] = None) -> None:
        cls._ensure_initialized()
        entry = cls._skill_feedback.setdefault(name, {"success": 0, "fail": 0, "notes": []})
        key = "success" if success else "fail"
        entry[key] += 1
        if metadata:
            entry["notes"].append(metadata)

    @classmethod
    def stats(cls) -> Dict[str, Any]:
        cls._ensure_initialized()
        return {name: {"success": data["success"], "fail": data["fail"], "notes": list(data["notes"])} for name, data in cls._skill_feedback.items()}


__all__ = ["SkillRegistry"]
