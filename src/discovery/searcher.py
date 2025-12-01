from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class ToolSearcher(ABC):
    """
    Finds candidate tools (APIs, libraries, repos) from external sources (GitHub, docs, etc.).
    """

    @abstractmethod
    def search(self, query: str) -> List[str]:
        raise NotImplementedError


class NoopToolSearcher(ToolSearcher):
    """
    v1 stub that returns no tools.
    """

    def search(self, query: str) -> List[str]:
        return []