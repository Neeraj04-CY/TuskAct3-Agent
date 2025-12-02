"""Custom exception hierarchy for the engine."""

from __future__ import annotations


class EikonError(RuntimeError):
    """Base exception for engine-specific failures."""


class CompletionError(EikonError):
    """Raised when completion metadata is missing or invalid."""


class WorkerExecutionError(EikonError):
    """Raised when a worker fails to complete work."""
