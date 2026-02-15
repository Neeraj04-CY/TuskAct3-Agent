from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ExecutionTrace


def _sanitize_segment(segment: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in segment)


class ExecutionTraceSerializer:
    """Persists ExecutionTrace payloads as JSON and keeps per-trace folders readable."""

    def __init__(self, *, ensure_ascii: bool = False) -> None:
        self.ensure_ascii = ensure_ascii

    def _ensure_trace_dir(self, trace: ExecutionTrace, *, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        trace_dir = directory / _sanitize_segment(trace.id)
        trace_dir.mkdir(parents=True, exist_ok=True)
        return trace_dir

    def build_path(self, trace: ExecutionTrace, *, directory: Path) -> Path:
        return self._ensure_trace_dir(trace, directory=directory) / "trace.json"

    def save(self, trace: ExecutionTrace, *, directory: Path) -> Path:
        file_path = self.build_path(trace, directory=directory)
        payload = trace.model_dump(mode="json")
        file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=self.ensure_ascii), encoding="utf-8")
        return file_path

    def load(self, path: Path | str) -> ExecutionTrace:
        file_path = Path(path)
        payload: Any = json.loads(file_path.read_text(encoding="utf-8"))
        return ExecutionTrace.model_validate(payload)


__all__ = ["ExecutionTraceSerializer"]
