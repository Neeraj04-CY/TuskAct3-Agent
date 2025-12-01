from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BrowserEngine(ABC):
    """
    Abstract browser automation engine.

    v1 can be a stub; v3 will implement Playwright.
    """

    @abstractmethod
    async def run_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class NoopBrowserEngine(BrowserEngine):
    """
    Stub implementation for environments where browser automation is disabled.
    """

    async def run_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "action": action,
            "params": params,
            "note": "Browser automation is disabled / not implemented."
        }