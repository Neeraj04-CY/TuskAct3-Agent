"""Shared type declarations for the EIKON Engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, TypedDict


class CompletionPayload(TypedDict, total=False):
    """Structure describing worker completion state."""

    complete: bool
    reason: str
    payload: Dict[str, Any]


class BrowserAction(TypedDict, total=False):
    """Declarative browser action specification."""

    action: str
    url: Optional[str]
    selector: Optional[str]
    value: Optional[str]
    name: Optional[str]
    timeout: Optional[int]


class BrowserWorkerResult(TypedDict):
    """Return payload from browser worker execution."""

    steps: List[Dict[str, Any]]
    screenshots: List[str]
    dom_snapshot: Optional[str]
    layout_graph: Optional[str]
    completion: CompletionPayload
    error: Optional[str]


class Worker(Protocol):
    """Protocol describing a generic worker."""

    async def run(self, description: str, prev_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task and return structured output."""


class CompletionAware(Protocol):
    """Protocol for objects that expose completion state."""

    def is_complete(self) -> bool:
        """Return True when the pipeline should terminate."""


AnyDict = Dict[str, Any]
