from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class SkillBuilder(ABC):
    """
    Builds a Skill File (JSON/YAML + optional Python wrapper) from parsed docs.
    """

    @abstractmethod
    def build(self, tool_description: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class NoopSkillBuilder(SkillBuilder):
    def build(self, tool_description: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": tool_description.get("name", "unknown_tool"),
            "source": tool_description.get("source", ""),
            "description": tool_description.get("description", ""),
            "version": "0.1.0",
            "rate_limit": "",
            "auth_methods": [],
            "endpoints": tool_description.get("endpoints", []),
            "example_usage": ""
        }