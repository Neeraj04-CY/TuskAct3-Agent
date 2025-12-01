from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class ApiEngine(ABC):
    """
    Abstract HTTP/API caller engine.
    """

    @abstractmethod
    async def call(self, method: str, url: str, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError


class SimpleApiEngine(ApiEngine):
    """
    Minimal, safe API engine. Does not manage auth yet.
    """

    async def call(self, method: str, url: str, **kwargs: Any) -> Dict[str, Any]:
        # TODO: implement with aiohttp/httpx and add safety controls
        return {
            "method": method,
            "url": url,
            "kwargs": kwargs,
            "note": "API engine not yet implemented."
        }