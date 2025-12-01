from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class CodeEngine(ABC):
    """
    Executes generated code in a controlled environment.
    """

    @abstractmethod
    async def run_code(self, language: str, code: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise NotImplementedError


class NoopCodeEngine(CodeEngine):
    """
    v1 stub that does not execute arbitrary code yet (for safety).
    """

    async def run_code(self, language: str, code: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return {
            "language": language,
            "code": code,
            "context": context or {},
            "note": "Code execution disabled / not implemented."
        }