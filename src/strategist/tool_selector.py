from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class ToolSelector(ABC):
    """
    Selects tools and skills to use for a given task or step.

    In v1 this can be rule-based and minimal;
    later versions can leverage skill metadata and Tool Discovery.
    """

    @abstractmethod
    def select_tools(self, task_description: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def select_skills(self, task_description: str) -> List[str]:
        raise NotImplementedError


class SimpleToolSelector(ToolSelector):
    """
    v1 tool selector using simple keyword heuristics.
    """

    def select_tools(self, task_description: str) -> List[str]:
        lowered = task_description.lower()
        tools: List[str] = []

        if "browser" in lowered or "website" in lowered or "scrape" in lowered:
            tools.append("browser_engine")

        if "api" in lowered or "http" in lowered:
            tools.append("api_engine")

        if "code" in lowered or "script" in lowered or "python" in lowered:
            tools.append("code_engine")

        return tools

    def select_skills(self, task_description: str) -> List[str]:
        # v1: no dynamic skills yet
        return []