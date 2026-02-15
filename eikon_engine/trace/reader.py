from __future__ import annotations

from pathlib import Path

from .models import ExecutionTrace
from .serializer import ExecutionTraceSerializer


def read_trace(path: Path | str) -> ExecutionTrace:
    """Load a persisted execution trace from disk."""

    serializer = ExecutionTraceSerializer()
    return serializer.load(path)


__all__ = ["read_trace"]
