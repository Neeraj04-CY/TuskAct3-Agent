from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict


class ToolScorer(ABC):
    """
    Scores tools by reliability, coverage, and performance.
    """

    @abstractmethod
    def score(self, tool_metadata: Dict[str, object]) -> float:
        raise NotImplementedError


class SimpleToolScorer(ToolScorer):
    """
    v1: constant score placeholder.
    """

    def score(self, tool_metadata: Dict[str, object]) -> float:
        return 0.5