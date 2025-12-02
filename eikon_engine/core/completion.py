"""Completion helpers shared across the engine."""

from __future__ import annotations

from typing import Any, Dict

from .types import CompletionPayload

DEFAULT_REASON = "worker did not specify a reason"


def build_completion(*, complete: bool, reason: str | None = None, payload: Dict[str, Any] | None = None) -> CompletionPayload:
    """Construct a normalized completion payload."""

    data: CompletionPayload = {
        "complete": complete,
        "reason": reason or DEFAULT_REASON,
        "payload": payload or {},
    }
    return data


def is_complete(result: Dict[str, Any]) -> bool:
    """Return True if the result dictionary indicates completion."""

    completion = result.get("completion")
    if not isinstance(completion, dict):
        return False
    return bool(completion.get("complete"))
