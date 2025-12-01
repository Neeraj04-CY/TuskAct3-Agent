from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class ApiTester(ABC):
    """
    Validates endpoints via test calls, checking schemas, auth, and rate limits.
    """

    @abstractmethod
    async def validate(self, endpoint_spec: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class NoopApiTester(ApiTester):
    async def validate(self, endpoint_spec: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "endpoint": endpoint_spec,
            "valid": False,
            "note": "API validation not implemented."
        }