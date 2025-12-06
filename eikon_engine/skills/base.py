from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class SkillBase(ABC):
    """Base class for Strategist skill plugins."""

    name: str = "skill"
    description: str = "Generic strategist plugin"
    version: str = "1.0.0"

    @abstractmethod
    def suggest_subgoals(self, state: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        """Return a list of subgoal suggestions based on the strategist state."""

    @abstractmethod
    def suggest_repairs(self, failure: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        """Return repair suggestions for the latest failure."""

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
        }


__all__ = ["SkillBase"]
